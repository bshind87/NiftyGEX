# Archive: exploratory analyses outside the proposal plan

Moved here on 2026-10-14 so the main code holds only the proposal path: the forecasting ladder (steps 0–6) and RQ1–RQ3. Nothing was deleted.

| Analysis | Code | Settings | Write-up |
|---|---|---|---|
| Stage A: HAC regressions of the next-day RV ratio on total GEX | `src/niftygex_eval/stage_a.py`, `scripts/run_stage_a.py` | `configs/eval_exploratory.yaml` (`stage_a`) | `docs/findings_t2.md` |
| Stage B: tercile classifiers, calibration, A/B/C/C1/C2 ablation, SHAP | `src/niftygex_eval/stage_b.py`, `src/niftygex_eval/metrics.py`, `scripts/run_stage_b.py`, `scripts/run_t2_ablation.py`, `scripts/plot_shap_interactions.py`, `scripts/write_findings_t2.py` | `stage_b` | `docs/findings_t2.md` |
| T3: max-GEX strike placebos (matched distance, listed strikes, continuous) | `src/niftygex_eval/t3_placebo.py`, `src/niftygex_eval/t3_listed.py`, `scripts/run_t3_placebo.py`, `scripts/run_t3_listed_placebo.py`, `scripts/write_findings_t3.py` | `t3_placebo`, `t3_listed_placebo` | `docs/findings_t3.md` |
| Holm families F1–F4 | `scripts/run_holm.py` | `multiple_testing` | both findings files |
| T1 pinning and T3 drift targets (data layer, removed 2026-10-14) | `src/niftygex/targets/pinning.py`, `src/niftygex/targets/intraday_drift.py`, `src/niftygex/io/official_closes.py` | former `snapshot_grid.t1_expiry_days` / `t3_daily`, `targets.t1` / `targets.t3`, `same_day_expiry_in_total_gex.t1` / `t3` in `configs/data.yaml` | `docs/schema_t1_t3.md` |

Tests are in `tests/test_eval_metrics.py`, `tests/test_t1.py` and `tests/test_t3_targets.py`. The method notes are in `docs/evaluation_exploratory.md`.

**Status.** The code here is a frozen copy.
- It is not on the import path and is not run by `pytest`, since `testpaths = ["tests"]`.
- Its last results were produced under data config hash `9fc36c738aa5`.
- Those results were moved to `outputs/archive/results/`. Their per-run directories stay in `outputs/results/runs/` and are listed in `index.jsonl`.

**Restore.** To rerun one of these analyses, move its files back to the same paths under `src/`, `scripts/` and `tests/`, and copy its section back into `configs/eval.yaml`.
- `stage_a.add_regressors` became `forecast/table.normalize_gex`.
- `stage_b.inner_walk` became `harness.inner_walk`.

The data layer no longer builds T1 or T3, which Stage B's T3 placebos and the T1 code need. Their last outputs are in `outputs/archive/data_targets/` (`t1/`, `t3/`) and `outputs/archive/reports/`. Restoring them means putting back the archived target modules, the config keys listed above, and the T1/T3 branches of the snapshot grid and the targets stage. `scripts/compute_targets_with_t1_t3.py` is the full pre-removal targets stage. The T1 wiring was never committed; the T3 branches are also in commit `b26d683`.
