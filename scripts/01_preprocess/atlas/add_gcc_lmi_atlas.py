# %% # Load packages

import os

# Plan 01-08 rework (2026-09-06): must be set before numpy/MDAnalysis/mdigest
# are imported anywhere in this process -- avoids BLAS/OMP thread
# oversubscription now that 16 worker processes already parallelize across
# the SLURM allocation. NOTE (2026-09-07): see process_atlas_cpu.py's
# corrected note -- the "fork-inherited lock deadlock" theory this comment
# originally cited was wrong; job 10203/10204's real cause was memory
# exhaustion under full concurrency (confirmed separately for this script:
# combined worker RSS grew from 51GB to 154GB in 6 minutes on a live run --
# see main()'s maxtasksperchild comment for the fix).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing

import h5py
from loguru import logger
from tqdm import tqdm

from planet_md import config
from planet_md.data.utils import update_h5_dataset
from planet_md.trajectory import compute_generalized_correlation_lmi

# Plan 01-08 rework: "spawn", not the default "fork" -- see
# planet_md/parallel.py's _SPAWN_CTX comment for the full rationale. Requires
# this script's dispatch logic to live behind `if __name__ == "__main__":`
# below (spawn re-imports this file as a plain module in every worker).
_SPAWN_CTX = multiprocessing.get_context("spawn")

# %% Define file paths
ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
ATLAS_H5 = ATLAS_PROCESSED_DATA_DIR / "atlas_processed.h5"
N_REPS = 3
RANDOM_STATE = 42
OVERWRITE_H5 = False


def _compute_gcc_lmi(job):
    pdb_code, rep, pdb_f, xtc_f = job
    gcorr = compute_generalized_correlation_lmi(pdb_f, xtc_f)
    return pdb_code, rep, gcorr


def main() -> None:
    logger.info("Defining file paths")
    pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
    pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]

    # Plan 01-08 rework (2026-09-06): this was the single largest contributor
    # to SLURM job 10202's ~7-day projection (~55s/rep, ~76% of per-batch
    # wall-clock, fully serial). GCC-LMI computation (mdigest DynCorr,
    # CPU-bound, no shared state between (protein, rep) pairs) runs in a
    # multiprocessing.Pool; the H5 write stays single-threaded in the parent
    # process (h5py is not safely writable from multiple processes at once).
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    jobs = [
        (
            pdb_f.stem,
            rep,
            pdb_f,
            ATLAS_DATA_DIR / pdb_f.stem[:2] / f"{pdb_f.stem}_prod_R{rep}_fit.xtc",
        )
        for pdb_f in pdb_files
        for rep in range(1, N_REPS + 1)
    ]

    logger.info(f"Computing GCC-LMI for {len(jobs)} (protein, rep) pairs across {n_jobs} worker processes")
    # Plan 01-08 rework (2026-09-07): confirmed live -- combined worker RSS
    # grew ~linearly (51GB -> 154GB over 6 minutes, aborted before it could
    # exhaust the shared node) even though a SINGLE largest-protein call
    # peaks at only ~4GB. That pattern (steady growth with task count, not a
    # few large outliers) points to per-task memory not being fully released
    # inside mdigest/MDAnalysis's DynCorr/Universe objects across repeated
    # calls within the same long-lived worker. maxtasksperchild recycles each
    # worker process after a bounded number of tasks, so any such
    # accumulation is reclaimed by the OS when the worker exits, instead of
    # growing for the life of the whole run.
    with _SPAWN_CTX.Pool(processes=n_jobs, maxtasksperchild=10) as pool, h5py.File(ATLAS_H5, "r+") as h5file:
        for pdb_code, rep, gcorr in tqdm(
            pool.imap_unordered(_compute_gcc_lmi, jobs),
            total=len(jobs),
            desc="Computing GCC-LMI",
        ):
            update_h5_dataset(h5file, f"{pdb_code}/R{rep}/gcc_lmi", gcorr, overwrite=True)


if __name__ == "__main__":
    main()
