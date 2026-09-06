# %% # Load packages
import gc
import sys

import mdtraj as md
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from planet_md import config
from planet_md.trajectory import (
    compute_autocorrelation,
    compute_contacts,
    compute_rmsf,
    normalize,
)

# %% Define file paths
logger.info("Defining file paths")

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
ATLAS_FOLDSEEK_CLUSTERS_FILE = (
    config.PROCESSED_DATA_DIR / "atlas/foldseek_atlas_0.2_cluster.tsv"
)

xtc_files = list(ATLAS_DATA_DIR.glob("*/*.xtc"))
pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]
N_REPS = 3
RANDOM_STATE = 42
OVERWRITE_H5 = False
I_START = int(sys.argv[1])
I_STOP = int(sys.argv[2])
# Plan 01-08 (full-dataset batched run): an optional 3rd CLI arg names this
# invocation's output dataset uniquely (e.g. "batch0003"). Without this, every
# per-batch invocation of this script under the mandatory streaming/batching
# architecture (Plan 01-07's checkpoint) would write to the same
# atlas_derivatives_v2_{I_START}_{I_STOP} directory whenever two batches
# happen to have the same size (I_START/I_STOP are always local indices into
# whatever's currently in RAW_DATA_DIR/atlas, i.e. just the current batch,
# since prior batches' raw files are deleted before the next download) --
# silently overwriting or crashing on Dataset.save_to_disk's
# directory-already-exists check. Defaults to the old "{I_START}_{I_STOP}"
# naming for backward compatibility with a single, non-batched invocation.
BATCH_LABEL = sys.argv[3] if len(sys.argv) > 3 else f"{I_START}_{I_STOP}"

# %% Define process function


def compute_trajectory_derivatives(pdb_id):
    # process = psutil.Process(os.getpid())
    # logger.info(f"Worker {os.getpid()} initial memory: {process.memory_info().rss / 1024 / 1024} MB")
    # logger.info(f"Starting batch of {len(pdb_id)}")
    pdb_code, rep = pdb_id
    logger.info(f"Processing {pdb_code}:{rep}")
    # logger.info(f"Worker {os.getpid()} memory before trajectory: {process.memory_info().rss / 1024 / 1024} MB")

    # Load trajectory
    xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
    pdb_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}.pdb"
    traj = md.load(str(xtc_f), top=pdb_f)
    # logger.info(f"Worker {os.getpid()} memory after load: {process.memory_info().rss / 1024 / 1024} MB")

    traj = normalize(traj, ca_only=True)
    rmsf = compute_rmsf(traj, normalized=True, ca_only=True)
    # logger.info(f"Worker {os.getpid()} memory after RMSF: {process.memory_info().rss / 1024 / 1024} MB")

    contacts = compute_contacts(
        traj, scheme="ca", ignore_nonprotein=True, normalized=True, ca_only=True
    )
    # logger.info(f"Worker {os.getpid()} memory after contacts: {process.memory_info().rss / 1024 / 1024} MB")

    ca_dist = contacts[0]
    autocorr = compute_autocorrelation(
        traj, precomputed_contacts=contacts, normalized=True, ca_only=True
    )
    # logger.info(f"Worker {os.getpid()} memory after autocorr: {process.memory_info().rss / 1024 / 1024} MB")

    r = {
        "pdb_code": pdb_code,
        "rep": rep,
        "xtc_file": xtc_f.stem,
        "pdb_file": pdb_f.stem,
        "rmsf": rmsf.copy(),
        "ca_dist": ca_dist.copy(),
        "autocorr": autocorr.copy(),
    }

    # Explicit cleanup
    del traj, contacts, rmsf, ca_dist, autocorr
    gc.collect()  # Force garbage collection
    # logger.info(f"Worker {os.getpid()} memory after cleanup: {process.memory_info().rss / 1024 / 1024} MB")
    # logger.info(f"Worker {os.getpid()} has {len(gc.get_objects())} total objects")

    return r


def compute_batched_trajectory_derivatives(pdb_id):
    # process = psutil.Process(os.getpid())
    # logger.info(f"Worker {os.getpid()} initial memory: {process.memory_info().rss / 1024 / 1024} MB")
    # logger.info(f"Starting batch of {len(pdb_id)}")

    results = []
    for pid in pdb_id:
        r = compute_trajectory_derivatives(pid)
        results.append(r)
    return results


# %% Define all reps
pdb_codes = [pdb_f.stem for pdb_f in pdb_files]
pdb_reps = [(pdb_code, rep) for pdb_code in pdb_codes for rep in range(1, N_REPS + 1)][
    I_START:I_STOP
]
TOTAL_JOBS = len(pdb_reps)

# %% Compute values in parallel
# N_JOBS = 50
# BATCH_SIZE = 1

# logger.info(f"Computing {TOTAL_JOBS} reps in parallel ({N_JOBS} workers, batch size {BATCH_SIZE})")
# results = parallel_pool(pdb_reps, compute_batched_trajectory_derivatives, n_jobs=N_JOBS, batch_size=BATCH_SIZE, report_every=1)
# logger.info(f"Finished processing {len(results)} reps")


# %% Invert Dicts
def invert_dict(l):
    """
    Convert from list of dicts to dict of lists, where the key is the joined "pdb_code" and "rep" keys of the inner dictionary
    """
    out_dict = {}
    keys = l[0].keys()
    for key in keys:
        out_dict[key] = [i[key] for i in l]
    return out_dict


# %% Just do serial to not deal with memory issues
results = []
logger.info(f"Computing {TOTAL_JOBS} reps")
for pdb_id in tqdm(pdb_reps):
    # results.append(compute_trajectory_derivatives(pdb_id))
    results.append(compute_trajectory_derivatives(pdb_id))
# %% # Create HuggingFace dataset

logger.info("Creating HuggingFace dataset")
ds = Dataset.from_dict(invert_dict(results))
logger.info(f"Dataset created with {len(ds)} samples")
out_path = ATLAS_PROCESSED_DATA_DIR / f"atlas_derivatives_v2_{BATCH_LABEL}"
logger.info(f"Saving dataset to {out_path}")
if out_path.exists():
    # Re-running the same batch label (e.g. after a crash) should overwrite
    # its own prior (possibly partial) output, not crash on
    # save_to_disk's directory-already-exists check.
    import shutil

    shutil.rmtree(out_path)
ds.save_to_disk(str(out_path))

# %% Join data sets from different initializations
# NOTE (Plan 01-08): the single-invocation join into a shared
# "atlas_derivatives_v2" was removed here -- under the mandatory batched
# streaming architecture (Plan 01-07's checkpoint), this script runs once per
# batch (see run_full_atlas_pipeline.py), so joining per-invocation would
# either crash (Dataset.save_to_disk refuses an existing directory) or
# silently clobber every prior batch's join with only the latest batch's data.
# The cross-batch join now happens exactly once, after all batches complete,
# in run_full_atlas_pipeline.py's join_all_batches() (or manually via
# `datasets.concatenate_datasets` over every atlas_derivatives_v2_batch*
# directory) -- matching Task 2 step 2's "once all chunk jobs complete, join
# every ... output" instruction, just performed by the orchestrating driver
# rather than by this per-chunk script itself.
logger.info(f"Batch {BATCH_LABEL} derivatives saved to {out_path} (cross-batch join happens once, after all batches, in run_full_atlas_pipeline.py)")
