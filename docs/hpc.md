# Exploratory neural runs on Explorer

All exploratory neural training (configs/eval_exploratory_v2.yaml: E1 neural parts, E2, E3 neural rungs and fresh
L = 12 references) runs on Northeastern's Explorer cluster. The Mac keeps code, CPU work (E3 data build, E3 classical
ladder, E1 residual LightGBM, combination, conditional IC), seed merging and reports. The pre-registered RQ1–RQ3
results stay the Mac runs. Course material on the cluster is in `docs/hpc_guide/`.

**Device rule.** Neural models compared with each other run on the same device and GPU type (`hpc/hpc.yaml`
`gpu_type`, default `v100-sxm2`, recorded in every run spec). Neural-vs-classical comparisons may mix devices.

## Layout

| What | Where |
|---|---|
| Code (a clean clone of the committed repository) | `/home/$USER/niftygex` |
| Conda environment | `/home/$USER/envs/niftygex` (`env/environment-cuda.yml` + the PyTorch CUDA wheel) |
| Data and outputs | `/scratch/$USER/niftygex/{data,outputs}` (`configs/paths.hpc.yaml`, `NIFTYGEX_PROFILE=hpc`) |
| Job scripts, logs, markers | `/scratch/$USER/niftygex/outputs/hpc/{jobs,logs,status}` |

Scratch is purged each semester: `hpc/pull_results.sh` copies results back to the Mac.

## Steps

1. **Commit** on the Mac (GitHub Desktop) and note the commit id.
2. **Code and environment** (login node, then a GPU job):
   ```bash
   ssh <user>@login.explorer.northeastern.edu
   curl -O https://raw.githubusercontent.com/bshind87/NiftyGEX/<commit>/hpc/sync_code.sh   # or copy it over
   bash sync_code.sh <commit>
   cd ~/niftygex && sbatch hpc/setup_env.sbatch      # builds the env on gpu-short, checks the driver and CUDA, runs the nn tests
   ```
   If the log says torch does not see the GPU, resubmit with an older CUDA wheel:
   `sbatch --export=ALL,TORCH_INDEX=https://download.pytorch.org/whl/cu124 hpc/setup_env.sbatch`.
3. **Data** (Mac, then a CPU job): `HPC_USER=<user> bash hpc/push_data.sh`, then on the cluster
   `sbatch hpc/verify_transfer.sbatch` (every file's size, rows and SHA-256 against the Mac manifests;
   `outputs/reports/transfer_check.json`).
4. **Job scripts**: `NIFTYGEX_PROFILE=hpc PYTHONPATH=src python hpc/make_jobs.py --gpu-type v100-sxm2` (or `auto`: h200 when
   `sinfo` shows idle H200 nodes). One script per experiment, model, horizon and seed; each trains all walks of
   one run with per-walk checkpoints.
5. **Calibration**: `NIFTYGEX_PROFILE=hpc PYTHONPATH=src python hpc/scheduler.py --once --waves calibration` (two 2-walk
   attention + GRU jobs on gpu-short: T2 at L = 12 and E3 h = 1). Each log prints the minutes and peak GPU memory
   per walk; compare with the Mac (87 min for the full 3-seed L = 12 run).
6. **Waves**: `sbatch hpc/scheduler.sbatch` keeps up to 8 jobs submitted on `gpu` (4 run at once) plus 1 on
   `courses-gpu`, in priority order (references, E1, E3, E2, L = 72 last), and resubmits a job stopped at the
   8-hour limit; it resumes from its checkpoints. To poll from the Mac instead:
   `ssh <user>@login... 'cd ~/niftygex && NIFTYGEX_PROFILE=hpc PYTHONPATH=src ~/envs/niftygex/bin/python hpc/scheduler.py --once'`.
   Remove a partition from `hpc/hpc.yaml` to use the others only (e.g. `courses-gpu` alone).
7. **Status**: `bash hpc/status.sh` (queue, accounting, done / failed markers). A crashed job writes a `.failed`
   marker and is not resubmitted until `hpc/scheduler.py --once --retry-failed`.
8. **Results** (Mac): `HPC_USER=<user> bash hpc/pull_results.sh` copies the results store, logs and markers to
   `outputs/hpc_pull/` and runs `scripts/merge_seeds.py`: each run's three seeds are averaged per row exactly as
   a local three-seed run, every walk is rescored, and the merged runs are listed where the E1–E3 test scripts read
   them (`neural_comparison_hpc.json`, `neural_comparison_window_24_hpc.json`, `exploratory_e{1,2,3}_runs.json`).

## Records

Every run spec has the commit, `git_dirty`, device and requested GPU type; every walk records the SLURM job id,
partition, host, GPU name and peak GPU memory (outside the spec, so a resubmitted job resumes the same run).
`outputs/hpc/submissions.tsv` logs each submission.

## Known issue: Apple-GPU memory on long local runs

On the Mac (MPS, 16 GB unified memory) GPU driver memory grows by about 1 GB per walk for attention models at
L = 24 and is never returned while the process lives: the backend caches memory for every new tensor shape, and
each walk adds new ones (the ragged last training batch, the ragged last prediction pass, accumulation pieces).
Window 24 failed twice this way (4 and 6 Nov). With constant shapes memory stays flat (4.2 GB over 26 walks), and
one process per walk (`scripts/run_isolated.py`) releases it at exit while reproducing the 4 Nov main run exactly
(walks 0–2, all predictions identical). The exploratory runs moved to the cluster instead.
