"""Test-suite report: runs pytest once and records the result per test file, with a summary figure.

Writes outputs/reports/tests.json (counts per file, every test's outcome, environment and git state),
outputs/reports/test_run.log (the full pytest output) and outputs/reports/figures/tests_summary.png.
"""
from __future__ import annotations

import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt

from niftygex.config import REPO_ROOT, load_config
from niftygex.io.parquet import write_report
from niftygex_eval.figures import GRID, SERIES, TEXT, TEXT_2, save

# What each test file covers, for the summary table.
AREAS = {"test_calendar": "Trading calendar", "test_cleaning": "Raw cleaning and duplicates", "test_config": "Configuration and paths",
         "test_pricing": "Forward, IV and Greeks", "test_gex": "GEX conventions", "test_selection": "Expiry selection",
         "test_regimes": "Regimes and lot sizes", "test_features": "Aggregate features", "test_targets": "Target (T2)",
         "test_leakage": "Leakage and availability", "test_stage_contracts": "Stage contracts and token history",
         "test_harness": "Walk-forward harness", "test_forecast": "Classical ladder and metrics",
         "test_multiple_testing": "RQ tests and Holm", "test_nn_dataset": "Neural inputs and scaler",
         "test_nn_models": "Neural models (acceptance)", "test_nn_trainer": "Neural trainer"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def outcome(case: ET.Element) -> str:
    for tag in ("failure", "error", "skipped"):
        if case.find(tag) is not None:
            return {"failure": "failed", "error": "error", "skipped": "skipped"}[tag]
    return "passed"


def figure(files: list[dict], totals: dict, stamp: str, directory: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 0.26 * (len(files) + 2) + 0.9))
    ax.axis("off")
    rows = [[f["area"], f["file"], str(f["passed"]), str(f["failed"] + f["error"]), str(f["skipped"])] for f in files]
    rows.append(["Total", "", str(totals["passed"]), str(totals["failed"] + totals["error"]), str(totals["skipped"])])
    table = ax.table(cellText=rows, colLabels=["Area", "File", "Passed", "Failed", "Skipped"], bbox=[0, 0, 1, 1],
                     colWidths=[0.36, 0.34, 0.1, 0.1, 0.1], cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=TEXT if r == 0 or r == len(rows) else TEXT_2, weight="bold" if r in (0, len(rows)) else "normal")
        if c >= 2:
            cell.set_text_props(ha="right")
        if r == 0:
            cell.set_facecolor("#f1f0ec")
    ok = totals["failed"] + totals["error"] == 0
    ax.set_title(f"{totals['passed']} passed, {totals['failed'] + totals['error']} failed, {totals['skipped']} skipped "
                 f"in {totals['seconds']:.1f} s", color=SERIES[2] if ok else SERIES[1], fontsize=12, weight="bold")
    ax.text(0, -0.02, f"PYTHONPATH=src pytest · {stamp}", transform=ax.transAxes, va="top", fontsize=7.5, color=TEXT_2)
    return save(fig, directory, "tests_summary")


def main() -> None:
    config = load_config()
    reports = Path(config["paths"]["reports_root"])
    junit = reports / "tests_junit.xml"
    started = time.time()
    run = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", f"--junitxml={junit}"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    seconds = time.time() - started
    (reports / "test_run.log").write_text(run.stdout + run.stderr)

    per_file: dict[str, dict] = defaultdict(lambda: {"passed": 0, "failed": 0, "error": 0, "skipped": 0})
    tests = []
    for case in ET.parse(junit).getroot().iter("testcase"):
        module = case.get("classname", "").split(".")[-1]
        result = outcome(case)
        per_file[module][result] += 1
        tests.append({"file": f"tests/{module}.py", "test": case.get("name"), "outcome": result, "seconds": float(case.get("time", 0))})
    files = [{"file": f"tests/{m}.py", "area": AREAS.get(m, m), **c} for m, c in sorted(per_file.items(), key=lambda kv: list(AREAS).index(kv[0]) if kv[0] in AREAS else 99)]
    totals = {k: sum(f[k] for f in files) for k in ("passed", "failed", "error", "skipped")} | {"tests": len(tests), "seconds": seconds}

    commit, dirty = git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain"))
    stamp = f"{datetime.now(ZoneInfo('Asia/Kolkata')):%Y-%m-%d %H:%M} IST · commit {commit}{' + uncommitted changes' if dirty else ''} · Python {platform.python_version()}"
    path = figure(files, totals, stamp, reports / "figures")
    write_report(config, "tests", {"exit_code": run.returncode, "totals": totals, "files": files, "tests": tests, "git_commit": commit,
                                   "git_dirty": dirty, "python": platform.python_version(), "figure": str(path.relative_to(REPO_ROOT)),
                                   "log": str((reports / "test_run.log").relative_to(REPO_ROOT))})
    print(f"{totals['passed']} passed, {totals['failed'] + totals['error']} failed, {totals['skipped']} skipped -> {path}")
    sys.exit(run.returncode)


if __name__ == "__main__":
    main()
