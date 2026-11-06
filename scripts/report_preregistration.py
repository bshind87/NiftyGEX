"""Pre-registration evidence: when the RQ1-RQ3 rules in configs/eval.yaml were committed, and the neural runs after them.

Reads git blame for the ``multiple_testing.families`` RQ lines, the results-store index for the neural runs, and each
neural run's recorded commit, where the RQ text must be identical. Writes outputs/reports/preregistration.json and
outputs/reports/figures/preregistration_rq_rules.png (the lines as in the editor, with their commit and dates).
Times are shown in the machine's local zone, as git and the terminal show them.
"""
from __future__ import annotations

import json
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from niftygex.config import REPO_ROOT, load_config
from niftygex.io.parquet import write_report
from niftygex_eval.figures import GRID, SERIES, TEXT, TEXT_2, save

CONFIG = "configs/eval.yaml"
FAMILIES = ("RQ1", "RQ2", "RQ3")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def families_at(commit: str) -> dict[str, str] | None:
    """The RQ family texts of configs/eval.yaml at ``commit`` (None when the file or section is missing)."""
    try:
        config = yaml.safe_load(git("show", f"{commit}:{CONFIG}"))
    except subprocess.CalledProcessError:
        return None
    families = (config.get("multiple_testing") or {}).get("families") or {}
    return {k: families.get(k) for k in FAMILIES}


def blame(line: int) -> dict:
    """Last commit that changed ``line`` of the config, with its author time."""
    out = git("blame", "--porcelain", "-L", f"{line},{line}", CONFIG).splitlines()
    info = {"commit": out[0].split()[0]}
    for row in out[1:]:
        key, _, value = row.partition(" ")
        if key in ("author-time", "summary"):
            info[key] = value
    info["committed_at"] = datetime.fromtimestamp(int(info.pop("author-time"))).astimezone()
    return info


def figure(lines: list[tuple[int, str]], commit: dict, first_run: dict, main_run: dict, unchanged: bool, directory: Path) -> Path:
    wrapped = []
    for number, text in lines:
        indent = " " * (len(text) - len(text.lstrip()))
        rule = text.lstrip().startswith(FAMILIES)
        parts = textwrap.wrap(text.strip(), 104, initial_indent=indent, subsequent_indent=indent + "     ") or [""]
        wrapped += [(number if i == 0 else None, part, rule) for i, part in enumerate(parts)]
    fig, ax = plt.subplots(figsize=(10, 0.19 * len(wrapped) + 1.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(len(wrapped) - 0.3, -0.7)
    ax.add_patch(plt.Rectangle((0, -0.7), 1, len(wrapped) + 0.4, color="#f6f5f1", zorder=0))
    for row, (number, part, rule) in enumerate(wrapped):
        if number is not None:
            ax.text(0.01, row, str(number), ha="left", va="center", family="monospace", fontsize=7.5, color=TEXT_2)
        ax.text(0.05, row, part, ha="left", va="center", family="monospace", fontsize=7.5, color=SERIES[0] if rule else TEXT)
    ax.set_title(f"{CONFIG}: RQ1–RQ3 rules, committed {commit['committed_at']:%Y-%m-%d %H:%M %Z} ({commit['commit'][:7]})",
                 loc="left", fontsize=10.5, color=TEXT)
    notes = (f"First neural run {first_run['at']:%Y-%m-%d %H:%M %Z} (run {first_run['run_id']}); main run {main_run['at']:%Y-%m-%d %H:%M %Z} "
             f"(commit {main_run['git_commit'][:7]}). RQ text {'identical in every neural run' if unchanged else 'CHANGED in some neural runs'}.")
    ax.text(0, len(wrapped) + 0.3, notes, ha="left", va="top", fontsize=8, color=SERIES[2] if unchanged else SERIES[1])
    return save(fig, directory, "preregistration_rq_rules")


def main() -> None:
    config = load_config()
    results = Path(config["paths"]["outputs_root"]) / "results"
    text = (REPO_ROOT / CONFIG).read_text().splitlines()
    start = next(i for i, line in enumerate(text, 1) if line.strip() == "families:")
    end = max(i for i, line in enumerate(text, 1) if line.strip().startswith(FAMILIES))
    commits = {blame(n)["commit"]: blame(n) for n in range(start, end + 1)}
    latest = max(commits.values(), key=lambda c: c["committed_at"])

    index = [json.loads(line) for line in (results / "index.jsonl").read_text().splitlines() if line.strip()]
    neural = sorted((r for r in index if r.get("stage") == "neural"), key=lambda r: r["created_at_utc"])
    current = families_at("HEAD")
    runs = []
    for r in neural:
        spec = json.loads((results / "runs" / r["run_id"] / "run.json").read_text())
        git_state = spec.get("spec", {})
        commit = git_state.get("git_commit")
        runs.append({"run_id": r["run_id"], "model": r["model"], "tag": git_state.get("tag"), "git_commit": commit, "git_dirty": git_state.get("git_dirty"),
                     "at": datetime.fromisoformat(r["created_at_utc"]).astimezone(), "rq_text_same_as_head": families_at(commit) == current})
    main_comparison = json.loads((results / "neural_comparison.json").read_text())["runs"]
    main_run = next(r for r in runs if r["run_id"] == main_comparison["attention_gru"])
    unchanged = all(r["rq_text_same_as_head"] for r in runs)
    first = runs[0]

    path = figure([(n, text[n - 1]) for n in range(start, end + 1)], latest, first, main_run, unchanged, Path(config["paths"]["reports_root"]) / "figures")
    write_report(config, "preregistration", {
        "config": CONFIG, "lines": [start, end], "families": current,
        "committed": {"commit": latest["commit"], "summary": latest.get("summary"), "at": latest["committed_at"].isoformat()},
        "first_neural_run": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in first.items()},
        "main_run": {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in main_run.items()},
        "committed_before_first_neural_run": latest["committed_at"] < first["at"],
        "rq_text_identical_in_every_neural_run": unchanged, "neural_runs": len(runs),
        "figure": str(path.relative_to(REPO_ROOT)),
    })
    print(f"RQ rules committed {latest['committed_at']:%Y-%m-%d %H:%M %Z} ({latest['commit'][:7]}); first neural run "
          f"{first['at']:%Y-%m-%d %H:%M %Z}; identical in all {len(runs)} neural runs: {unchanged} -> {path}")


if __name__ == "__main__":
    main()
