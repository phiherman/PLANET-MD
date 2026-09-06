#!/usr/bin/bash
# Plan 01-08 (full-dataset ATLAS run): generalized from a single hardcoded
# `process_atlas_cpu.py 3000 4000` chunk (Flatiron Institute's original
# template, partition `genx`/`a100-40gb` -- neither exists on this cluster,
# confirmed via `sinfo` during Plan 01-07's checkpoint) into a submission
# wrapper for the full batched pipeline driver
# (run_full_atlas_pipeline.py), which internally loops over ~300-protein
# batches (download -> embed -> 3Di -> RMSF/CA-dist/autocorr -> GCC-LMI ->
# H5 write -> verify -> delete raw) rather than needing one SLURM job per
# chunk: `staging-server` is a single node (confirmed via `scontrol show
# node stagingenv`: 48 CPUs, 257GB RAM, 1 GPU split into 4x MIG 1g.24gb
# instances, partitions `gpu`/`shared`) -- splitting the sequential
# per-protein pipeline across multiple SLURM jobs on the same node would not
# add parallelism, only scheduling overhead. One long-running job (partition
# `gpu`, `--gres=gpu:1` for the ESM3 embedding steps) covers the whole run,
# which is multi-day, dominated by network transfer from dsimb.inserm.fr
# (~4-5 MB/s observed per connection) and per-protein GPU/CPU compute --
# `--time` is left unset since both partitions report TIMELIMIT=infinite.
#SBATCH --job-name=atlas_full_pipeline
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

PLANET_MD_DIR=${PLANET_MD_DIR:-/gpfs/projects/immunochem/PLANET-MD}
cd "$PLANET_MD_DIR"
source "$PLANET_MD_DIR/.conda/etc/profile.d/conda.sh"
conda activate planet_md

# --batch-size/--min-free-gb/--download-workers/--max-batches are all
# overridable via env vars at submission time (`sbatch --export=...`) without
# editing this file; defaults match Plan 01-07's mandated ~300-protein batch
# size and 500GB storage-safety floor.
python scripts/01_preprocess/atlas/run_full_atlas_pipeline.py \
    --batch-size "${ATLAS_BATCH_SIZE:-300}" \
    --min-free-gb "${ATLAS_MIN_FREE_GB:-500}" \
    --download-workers "${ATLAS_DOWNLOAD_WORKERS:-8}" \
    ${ATLAS_MAX_BATCHES:+--max-batches "$ATLAS_MAX_BATCHES"}
