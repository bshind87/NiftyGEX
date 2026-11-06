"""HPC tooling: job list, SLURM scripts, submission limits, seed merging (exactly as a local multi-seed run),
transfer verification, per-seed settings and the device override."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from niftygex.config import REPO_ROOT
from niftygex_eval.hpc_jobs import Job, job_list, plan_submissions, render_script
from niftygex_eval.merge import group_key, merge_predictions, merge_walk_metrics, merged_spec, seed_of

SETTINGS = {"resources": {"mem": "32G", "cpus_per_task": 4, "time": "08:00:00"}, "calibration": {"partition": "gpu-short", "time": "02:00:00", "max_submitted": 2},
            "scratch_dir": "/scratch/u/niftygex", "code_dir": "/home/u/niftygex", "env_dir": "/home/u/envs/niftygex",
            "modules": ["explorer", "anaconda3/2024.06", "cuda/12.1.1"], "job_prefix": "ngx_"}
PARTITIONS = [{"name": "gpu", "max_running": 4, "max_submitted": 8}, {"name": "courses-gpu", "max_running": 1, "max_submitted": 1}]


def test_job_list_covers_every_run_once_per_seed() -> None:
    jobs = job_list([11, 12, 13], [1, 2])
    names = [j.name for j in jobs]
    assert len(names) == len(set(names)) == 2 + 9 + 12 + 18 + 9 + 3
    assert [j.wave for j in jobs][:2] == ["calibration", "calibration"] and jobs[-1].wave == "e2_l72"
    for job in jobs:
        script = job.command.split()[0]
        assert (REPO_ROOT / script).exists(), script
        assert job.calibration or "--seed" in job.command


def test_job_scripts_fix_the_gpu_type_and_leave_the_partition_to_the_scheduler() -> None:
    text = render_script(Job("e3_attention_gru_1h_s11", "e3", "scripts/run_intraday.py --horizon 1 --models attention_gru --seed 11"), SETTINGS, "v100-sxm2")
    assert "#SBATCH --gres=gpu:v100-sxm2:1" in text and "--partition" not in text and "#SBATCH --time=08:00:00" in text
    assert "NIFTYGEX_GPU_TYPE=v100-sxm2" in text and "NIFTYGEX_DEVICE=cuda" in text and "module load explorer anaconda3/2024.06 cuda/12.1.1" in text
    assert "e3_attention_gru_1h_s11.done" in text and "e3_attention_gru_1h_s11.failed" in text
    assert "#SBATCH --time=02:00:00" in render_script(Job("cal", "calibration", "x", True), SETTINGS, "v100-sxm2")


def test_plan_respects_partition_limits_and_skips_finished_jobs() -> None:
    jobs = [Job(f"j{i}", "e3", "x") for i in range(12)] + [Job("cal", "calibration", "x", True)]
    plan, exhausted = plan_submissions(jobs, queued={"j0": "gpu"}, done={"j1"}, failed={"j2"}, attempts={"j3": 6},
                                       partitions=PARTITIONS, calibration=SETTINGS["calibration"], max_attempts=6, queue_counts={"gpu": 1})
    where = {job.name: partition for job, partition in plan}
    assert exhausted == ["j3"] and not {"j0", "j1", "j2", "j3"} & set(where)
    assert list(where.values()).count("gpu") == 7 and list(where.values()).count("courses-gpu") == 1   # 8 submitted on gpu, 1 on courses-gpu
    assert where["cal"] == "gpu-short"
    alone, _ = plan_submissions(jobs[:3], {}, set(), set(), {}, PARTITIONS[1:], SETTINGS["calibration"], 6, {})
    assert [p for _, p in alone] == ["courses-gpu"]


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
