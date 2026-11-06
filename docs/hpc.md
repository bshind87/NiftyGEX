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
| Code (a clean clone of the public repository at a commit) | `/home/shinde.b/niftygex` |
| Conda environment | `/home/shinde.b/envs/niftygex` (`env/environment-cuda.yml` + the PyTorch CUDA wheel) |
| Data and outputs (private: `go-rwx`, `umask 077` in every job) | `/scratch/shinde.b/niftygex/{data,outputs}` (`configs/paths.hpc.yaml`, `NIFTYGEX_PROFILE=hpc`) |
| Job scripts, logs, markers, submission log | `/scratch/shinde.b/niftygex/outputs/hpc/{jobs,logs,status,submissions.tsv}` |

**Clean environment.** The Explorer login shell activates an unrelated conda env from `~/.bashrc`, and SLURM copies
the submitting shell's environment into jobs. Every job script and remote command therefore loads the modules,
deactivates every inherited conda env, unsets `PYTHONPATH`/`PYTHONHOME`, sets `PYTHONNOUSERSITE=1`, activates
`/home/shinde.b/envs/niftygex` explicitly and checks that `python` comes from it; nothing depends on another env.

**Code access.** The repository is private: `hpc/sync_code.sh` clones `git@github-niftygex:bshind87/NiftyGEX.git`
with a read-only deploy key (ssh host alias `github-niftygex` on Explorer).

**Data transfer** goes through RC's transfer node (same `/home` and `/scratch`); rsync on the login node is cut off
after about a minute. Add to `~/.ssh/config` on the Mac:

```
Host explorer-xfer
    HostName xfer.discovery.neu.edu
    User shinde.b
    IdentityFile ~/.ssh/shinde.b-20261027T140307Z
    IdentitiesOnly yes
```

Commands (`hpc/remote.sh`) use the ssh host alias `explorer`; data copies (`hpc/push_data.sh`, `hpc/pull_results.sh`) use `explorer-xfer`.
Scratch is purged each semester: `hpc/pull_results.sh` copies results back to the Mac.

## Steps

1. **Commit and push** on the Mac (GitHub Desktop); note the commit id.
2. **Code**: `bash hpc/remote.sh sync <commit>` (clone or update, refuse a dirty tree, create the private scratch layout).
3. **Environment**: `bash hpc/remote.sh setup` builds the env on a CPU node (`short`; an install on a GPU node leaves
   the GPU idle and RC cancels idle GPU jobs) and runs the CPU tests, then `hpc/cuda_check.sbatch` checks the driver,
   `torch.cuda` and a matmul on a v100-sxm2 (gpu-short). If torch does not see the GPU, resubmit with an older wheel:
   `ssh explorer 'cd ~/niftygex && sbatch --export=ALL,TORCH_INDEX=https://download.pytorch.org/whl/cu124 hpc/setup_env.sbatch'`.
4. **Data**: `bash hpc/push_data.sh` (rsync, then `chmod -R go-rwx`), then `bash hpc/remote.sh verify` (every file's size,
   rows and SHA-256 against the Mac manifests; `outputs/reports/transfer_check.json`).
5. **Job scripts**: `bash hpc/remote.sh jobs v100-sxm2` (53 scripts; one per experiment, model, horizon and seed).
6. **Calibration**: `bash hpc/remote.sh scheduler --waves calibration` (two 2-walk attention + GRU jobs: T2 at L = 12 and
   E3 h = 1). Logs print minutes and peak GPU memory per walk. The measured walk minutes go into `hpc/estimates.yaml`
   (committed), then step 5 again so each script knows its walk length.
7. **Waves**: `bash hpc/remote.sh loop` submits `hpc/scheduler.sbatch` (a CPU job polling every 5 minutes); or run
   `bash hpc/remote.sh scheduler` from the Mac now and then. Routing (`hpc/hpc.yaml`):
   - short jobs (under 2 h: references, E1 neural, E2 L = 24 and multi-day, E3 DeepSets and attention) go to
     **gpu-short** (2 running / 4 submitted, `--time 01:55:00`; RC also lists a 1-GPU-per-user limit there);
   - long jobs (E3 attention + GRU, E2 L = 72) are first queued on **gpu** (4 / 8, 8 h) for the night;
   - **courses-gpu** (1 job, 8 h) takes any job whose `sbatch --test-only` start is earlier there;
   - a long job still pending on gpu moves to chained **gpu-short pieces** (1 h 50 min) when gpu-short has a free slot,
     no short job is waiting, its walks fit a piece, and the piece starts at least 30 min sooner;
   - every job stops starting walks 8 minutes before its time limit and is resubmitted; it resumes from its walk
     checkpoints. A crash writes a `.failed` marker and stops that job until `--retry-failed`.
8. **Status**: `bash hpc/remote.sh status`.
9. **Results** (Mac): `bash hpc/pull_results.sh` copies the results store, logs, markers and reports to
   `outputs/hpc_pull/` and runs `scripts/merge_seeds.py`: each run's three seeds are averaged per row exactly as a
   local three-seed run, every walk is rescored, and the merged runs are listed where the E1–E3 test scripts read
   them (`neural_comparison_hpc.json`, `neural_comparison_window_24_hpc.json`, `exploratory_e{1,2,3}_runs.json`).

## Records

Every run spec has the commit, `git_dirty`, device and GPU type; every walk records the SLURM job id, partition, node,
GPU name and peak GPU memory (outside the spec, so a resubmitted job resumes the same run), and the run summary
lists them. `outputs/hpc/submissions.tsv` logs each submission and move with its partition, time limit and commit.

## Known issue: Apple-GPU memory on long local runs

On the Mac (MPS, 16 GB unified memory) GPU driver memory grows by about 1 GB per walk for attention models at
L = 24 and is never returned while the process lives: the backend caches memory for every new tensor shape, and
each walk adds new ones (the ragged last training batch, the ragged last prediction pass, accumulation pieces).
Window 24 failed twice this way (4 and 6 Nov). With constant shapes memory stays flat (4.2 GB over 26 walks), and
one process per walk (`scripts/run_isolated.py`) releases it at exit while reproducing the 4 Nov main run exactly
(walks 0–2, all predictions identical). The exploratory runs moved to the cluster instead.
