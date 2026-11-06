"""HPC tooling: job list, SLURM scripts, submission limits, seed merging (exactly as a local multi-seed run),
transfer verification, per-seed settings and the device override."""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from niftygex.config import REPO_ROOT
from niftygex_eval.hpc_jobs import Job, Queued, job_list, plan, render_script
from niftygex_eval.merge import group_key, merge_predictions, merge_walk_metrics, merged_spec, seed_of

SETTINGS = {"resources": {"mem": "32G", "cpus_per_task": 4}, "scratch_dir": "/scratch/u/niftygex", "code_dir": "/home/u/niftygex",
            "env_dir": "/home/u/envs/niftygex", "modules": ["explorer", "anaconda3/2024.06", "cuda/12.1.1"], "job_prefix": "ngx_",
            "deadline_margin_minutes": 8, "max_attempts": 12,
            "partitions": {"gpu-short": {"max_running": 2, "max_submitted": 4, "time": "01:55:00"},
                           "gpu": {"max_running": 4, "max_submitted": 8, "time": "08:00:00"},
                           "courses-gpu": {"max_running": 1, "max_submitted": 1, "time": "08:00:00"}},
            "routing": {"short": ["gpu-short", "courses-gpu"], "long": ["gpu", "courses-gpu"], "chain_on": "gpu-short",
                        "chain_time": "01:50:00", "chain_max_walk_minutes": 80, "move_if_earlier_by_minutes": 30}}
NOW = datetime(2026, 11, 6, 13, 0)


def starts(table: dict):
    """A stand-in for sbatch --test-only: start time by partition, in hours from now."""
    return lambda job, partition, time_limit: NOW + timedelta(hours=table[partition]) if partition in table else None


def test_job_list_covers_every_run_once_per_seed() -> None:
    jobs = job_list([11, 12, 13], [1, 2])
    names = [j.name for j in jobs]
    assert len(names) == len(set(names)) == 2 + 9 + 12 + 18 + 9 + 3
    assert [j.wave for j in jobs][:2] == ["calibration", "calibration"] and jobs[-1].wave == "e2_l72"
    long = {j.name for j in jobs if j.kind == "long"}
    assert long == {f"e3_attention_gru_{h}h_s{s}" for h in (1, 2) for s in (11, 12, 13)} | {f"e2_window72_s{s}" for s in (11, 12, 13)}
    for job in jobs:
        assert (REPO_ROOT / job.command.split()[0]).exists(), job.command
        assert job.wave == "calibration" or "--seed" in job.command


def test_job_scripts_fix_the_gpu_type_keep_data_private_and_stop_before_the_deadline() -> None:
    job = Job("e3_attention_gru_1h_s11", "e3", "scripts/run_intraday.py --horizon 1 --models attention_gru --seed 11", "long", "e3_attention_gru")
    text = render_script(job, SETTINGS, "v100-sxm2", 42.0)
    assert "#SBATCH --gres=gpu:v100-sxm2:1" in text and "--partition" not in text and "#SBATCH --time" not in text
    assert "umask 077" in text and "NIFTYGEX_GPU_TYPE=v100-sxm2" in text and "NIFTYGEX_DEVICE=cuda" in text
    assert "NIFTYGEX_DEADLINE=$(( $(date -d \"$end\" +%s) - 480 ))" in text and "NIFTYGEX_WALK_MINUTES=42.0" in text
    assert "module load explorer anaconda3/2024.06 cuda/12.1.1" in text and "e3_attention_gru_1h_s11.done" in text
    assert "NIFTYGEX_WALK_MINUTES" not in render_script(job, SETTINGS, "v100-sxm2", None)


def test_short_jobs_go_to_gpu_short_and_long_jobs_to_gpu_unless_courses_gpu_starts_sooner() -> None:
    jobs = [Job("s1", "e1", "x"), Job("l1", "e3", "x", "long", "g")]
    actions, _ = plan(jobs, {}, set(), set(), {}, SETTINGS, {}, starts({"gpu-short": 0.5, "gpu": 15, "courses-gpu": 3}), NOW)
    assert [(a.job.name, a.partition, a.time) for a in actions] == [("s1", "gpu-short", "01:55:00"), ("l1", "courses-gpu", "08:00:00")]
    actions, _ = plan(jobs, {}, set(), set(), {}, SETTINGS, {}, starts({"gpu-short": 0.5, "gpu": 2, "courses-gpu": 3}), NOW)
    assert [a.partition for a in actions] == ["gpu-short", "gpu"]
    many = [Job(f"s{i}", "e1", "x") for i in range(7)]
    actions, _ = plan(many, {}, set(), set(), {}, SETTINGS, {}, starts({"gpu-short": 0.5, "courses-gpu": 1}), NOW)
    assert [a.partition for a in actions].count("gpu-short") == 4 and [a.partition for a in actions].count("courses-gpu") == 1


def test_long_jobs_chain_on_gpu_short_only_when_it_is_free_and_their_walks_fit() -> None:
    long = Job("l1", "e3", "x", "long", "g")
    pending = {"l1": Queued("gpu", "PENDING", "900", NOW + timedelta(hours=15))}
    move, _ = plan([long], pending, set(), set(), {}, SETTINGS, {"g": 40}, starts({"gpu-short": 0.5}), NOW)
    assert [(a.kind, a.partition, a.time, a.cancel) for a in move] == [("move", "gpu-short", "01:50:00", "900")]
    too_long, _ = plan([long], pending, set(), set(), {}, SETTINGS, {"g": 95}, starts({"gpu-short": 0.5}), NOW)
    unknown, _ = plan([long], pending, set(), set(), {}, SETTINGS, {}, starts({"gpu-short": 0.5}), NOW)
    busy, _ = plan([long, Job("s1", "e1", "x")], pending, set(), set(), {}, SETTINGS, {"g": 40}, starts({"gpu-short": 0.5}), NOW)
    assert too_long == [] and unknown == [] and [a.job.name for a in busy] == ["s1"]   # a waiting short job keeps the slot
    table = starts({"gpu-short": 0.5, "gpu": 15, "courses-gpu": 12})
    fresh, _ = plan([long], {}, set(), set(), {}, SETTINGS, {"g": 40}, table, NOW)
    assert [(a.partition, a.time) for a in fresh] == [("courses-gpu", "08:00:00")]          # first queued for the night
    resumed, _ = plan([long], {}, set(), set(), {"l1": 1}, SETTINGS, {"g": 40}, table, NOW)
    assert [(a.partition, a.time) for a in resumed] == [("gpu-short", "01:50:00")]           # continues as a chained piece


def test_plan_skips_finished_queued_and_exhausted_jobs() -> None:
    jobs = [Job(f"j{i}", "e1", "x") for i in range(5)]
    queued = {"j0": Queued("gpu-short", "RUNNING", "1")}
    actions, exhausted = plan(jobs, queued, {"j1"}, {"j2"}, {"j3": 12}, SETTINGS, {}, starts({"gpu-short": 0.5}), NOW)
    assert exhausted == ["j3"] and [a.job.name for a in actions] == ["j4"]


def test_frame_differences_allow_only_float_rounding() -> None:
    from niftygex.io.parquet import frame_differences

    a = pl.DataFrame({"x": [1.0, 2.0, None], "k": ["a", "b", "c"]})
    assert frame_differences(a, a.with_columns(pl.col("x") + 1e-16))["problems"] == []
    assert frame_differences(a, a.with_columns(pl.col("x") + 1e-6))["problems"] == ["x"]
    assert frame_differences(a, a.with_columns(pl.lit("z").alias("k")))["problems"] == ["k"]


def test_runner_stops_before_a_walk_it_cannot_finish(tmp_path, monkeypatch) -> None:
    from test_exploratory import daily_table

    from niftygex_eval.forecast.baselines import Linear
    from niftygex_eval.forecast.runner import MORE_WALKS, run_walks
    from niftygex_eval.harness import build_walks

    table = daily_table()
    walks = build_walks(table["cycle"].unique().to_list(), test_cycles=8, embargo_cycles=1, min_train_cycles=20)
    monkeypatch.setenv("NIFTYGEX_DEADLINE", str(time.time() + 600))
    monkeypatch.setenv("NIFTYGEX_WALK_MINUTES", "30")            # a walk is expected to take longer than the time left
    with pytest.raises(SystemExit) as stop:
        run_walks(table, walks, lambda: Linear("ols", ["a"], []), "y", inner_validation_cycles=4, inner_embargo_cycles=1, checkpoint=tmp_path)
    assert stop.value.code == MORE_WALKS and not list(tmp_path.glob("walk_*"))
    monkeypatch.setenv("NIFTYGEX_WALK_MINUTES", "0")
    predictions, _ = run_walks(table, walks, lambda: Linear("ols", ["a"], []), "y", inner_validation_cycles=4, inner_embargo_cycles=1, checkpoint=tmp_path)
    assert predictions.height > 0


torch = pytest.importorskip("torch")


def test_merging_seed_runs_equals_one_run_with_all_seeds() -> None:
    from test_nn_trainer import NEURAL, factory, synthetic  # stand-in model, CPU (deterministic)

    from niftygex_eval.forecast.runner import run_walks
    from niftygex_eval.harness import build_walks
    from niftygex_eval.nn.trainer import NeuralForecaster

    table, inputs = synthetic()
    walks = build_walks(table["cycle"].unique().to_list(), test_cycles=8, embargo_cycles=1, min_train_cycles=30)

    def run(seeds):
        neural = {**NEURAL, "training": {**NEURAL["training"], "refit": False, "seeds": seeds}}
        return run_walks(table, walks, lambda: NeuralForecaster("stand_in", inputs, neural, factory=factory), "y",
                         inner_validation_cycles=6, inner_embargo_cycles=1)

    together = run([1, 2])
    separate = {seed: run([seed]) for seed in (1, 2)}
    predictions = merge_predictions({s: p for s, (p, _) in separate.items()})
    assert np.array_equal(predictions["prediction"].to_numpy(), together[0].sort("snapshot_at")["prediction"].to_numpy())
    per_walk = merge_walk_metrics({s: w for s, (_, w) in separate.items()}, predictions)
    for column in ("ic", "rmse", "qlike", "stop_epoch_mean", "stop_epoch_max"):
        assert per_walk[column].to_numpy() == pytest.approx(together[1][column].to_numpy()), column


def test_seed_runs_group_by_everything_but_the_seed() -> None:
    spec = {"stage": "neural", "model": "attention_gru", "tag": "window_24_hpc_seed11", "training": {"lr": 0.001, "seeds": [11]},
            "git_commit": "abc", "git_dirty": False}
    other = {**spec, "tag": "window_24_hpc_seed12", "training": {"lr": 0.001, "seeds": [12]}}
    assert group_key(spec) == group_key(other) and seed_of(spec) == 11
    assert group_key({**other, "window": 72}) != group_key(spec)
    merged = merged_spec({11: spec, 12: other}, {11: "r11", 12: "r12"})
    assert merged["tag"] == "window_24_hpc" and merged["training"]["seeds"] == [11, 12] and merged["merged_from"] == {"11": "r11", "12": "r12"}
    with pytest.raises(ValueError):
        merged_spec({11: spec, 12: {**other, "git_commit": "def"}}, {11: "a", 12: "b"})


def test_transfer_check_catches_missing_and_changed_files(tmp_path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("verify_transfer", REPO_ROOT / "scripts" / "verify_transfer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = {"data_root": tmp_path / "data", "outputs_root": tmp_path / "outputs"}
    target = tmp_path / "data" / "features" / "x.parquet"
    target.parent.mkdir(parents=True)
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(target)
    record = {"path": "data_root/features/x.parquet", "size_bytes": target.stat().st_size, "rows": 3}
    assert module.resolve(record["path"], paths) == target
    assert module.resolve("/Users/me/repo/data/raw/futures/f.csv.gz", paths) == tmp_path / "data" / "raw" / "futures" / "f.csv.gz"
    assert module.check(record, target, checksums=True) is None
    assert module.check({**record, "rows": 4}, target, checksums=False) == "rows 3 != 4"
    assert module.check(record, tmp_path / "nope.parquet", checksums=False) == "missing"
    assert module.resolve("/Users/me/repo/docs/report.md", paths) is None


def test_seed_and_device_settings(monkeypatch) -> None:
    from niftygex_eval.compute import spec_extras, with_seed
    from niftygex_eval.nn.trainer import resolve_device

    neural = {"training": {"seeds": [11, 12, 13], "lr": 0.001}}
    assert with_seed(neural, 12)["training"]["seeds"] == [12] and with_seed(neural, None) is neural
    with pytest.raises(SystemExit):
        with_seed(neural, 7)
    monkeypatch.delenv("NIFTYGEX_GPU_TYPE", raising=False)
    assert spec_extras() == {}
    monkeypatch.setenv("NIFTYGEX_GPU_TYPE", "v100-sxm2")
    assert spec_extras() == {"gpu_type": "v100-sxm2"}
    monkeypatch.setenv("NIFTYGEX_DEVICE", "cuda")
    assert resolve_device("cpu") == "cuda" and resolve_device("mps") == "cuda"
    monkeypatch.delenv("NIFTYGEX_DEVICE")
    assert resolve_device("cpu") == "cpu"
    assert json.loads(json.dumps(spec_extras())) == {"gpu_type": "v100-sxm2"}
    assert Path(REPO_ROOT / "configs" / "paths.hpc.yaml").exists()
