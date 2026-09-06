# %% # Load packages

import os
from multiprocessing import Pool

import h5py
from loguru import logger
from tqdm import tqdm

from planet_md import config
from planet_md.data.utils import update_h5_dataset
from planet_md.trajectory import compute_generalized_correlation_lmi

# %% Define file paths
logger.info("Defining file paths")
ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
ATLAS_H5 = ATLAS_PROCESSED_DATA_DIR / "atlas_processed.h5"

xtc_files = list(ATLAS_DATA_DIR.glob("*/*.xtc"))
pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]
N_REPS = 3
RANDOM_STATE = 42
OVERWRITE_H5 = False
# Plan 01-08 rework (2026-09-06): this was the single largest contributor to
# SLURM job 10202's ~7-day projection (~55s/rep, ~76% of per-batch wall-clock,
# fully serial). GCC-LMI computation (mdigest DynCorr, CPU-bound, no shared
# state between (protein, rep) pairs) runs in a multiprocessing.Pool; the H5
# write stays single-threaded in the main process (h5py is not safely
# writable from multiple processes at once).
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

# %% Compute derivatives and store


def _compute_gcc_lmi(job):
    pdb_code, rep, pdb_f, xtc_f = job
    gcorr = compute_generalized_correlation_lmi(pdb_f, xtc_f)
    return pdb_code, rep, gcorr


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

logger.info(f"Computing GCC-LMI for {len(jobs)} (protein, rep) pairs across {N_JOBS} worker processes")
# Pool created before the H5 file is opened, so forked workers never inherit
# an open HDF5 file handle -- only the parent process ever touches ATLAS_H5.
with Pool(processes=N_JOBS) as pool, h5py.File(ATLAS_H5, "r+") as h5file:
    for pdb_code, rep, gcorr in tqdm(
        pool.imap_unordered(_compute_gcc_lmi, jobs),
        total=len(jobs),
        desc="Computing GCC-LMI",
    ):
        update_h5_dataset(h5file, f"{pdb_code}/R{rep}/gcc_lmi", gcorr, overwrite=True)
