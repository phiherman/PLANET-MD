# %% # Load packages
import os

# Plan 01-08 rework (2026-09-06): must be set before numpy/mdtraj/BLAS are
# imported anywhere in this process. With 16 worker processes already
# parallelizing across the SLURM allocation's CPUs, each worker ALSO letting
# OpenBLAS/MKL/OMP spin up its own internal thread pool oversubscribes the
# machine (16 workers x N BLAS threads > allocated CPUs) and -- more
# seriously -- leaves live background threads in the parent process at the
# moment multiprocessing.Pool forks workers. A thread frozen mid-lock by
# fork() is inherited by every child in that locked state forever, which is
# what caused SLURM job 10203 to deadlock (confirmed: every worker showed
# wchan=futex_wait_queue with zero CPU progress). Pinning every BLAS/OMP
# backend to 1 thread removes the extra background threads entirely.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import sys

import mdtraj as md
from datasets import Dataset
from loguru import logger

from planet_md import config
from planet_md.parallel import parallel_pool
from planet_md.trajectory import (
    compute_autocorrelation,
    compute_contacts,
    compute_rmsf,
    normalize,
)

# %% Define file paths
ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
ATLAS_FOLDSEEK_CLUSTERS_FILE = (
    config.PROCESSED_DATA_DIR / "atlas/foldseek_atlas_0.2_cluster.tsv"
)
N_REPS = 3
RANDOM_STATE = 42
OVERWRITE_H5 = False

# %% Define process function (module-level: must be importable/picklable by
# spawned worker processes without triggering main()'s side effects below --
# see the `if __name__ == "__main__"` guard, required by the "spawn" start
# context parallel_pool() now uses).


def compute_trajectory_derivatives(pdb_id):
    pdb_code, rep = pdb_id
    logger.info(f"Processing {pdb_code}:{rep}")

    # Load trajectory
    xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
    pdb_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}.pdb"
    traj = md.load(str(xtc_f), top=pdb_f)

    traj = normalize(traj, ca_only=True)
    rmsf = compute_rmsf(traj, normalized=True, ca_only=True)

    contacts = compute_contacts(
        traj, scheme="ca", ignore_nonprotein=True, normalized=True, ca_only=True
    )

    ca_dist = contacts[0]
    autocorr = compute_autocorrelation(
        traj, precomputed_contacts=contacts, normalized=True, ca_only=True
    )

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

    return r


def compute_batched_trajectory_derivatives(pdb_id):
    results = []
    for pid in pdb_id:
        r = compute_trajectory_derivatives(pid)
        results.append(r)
    return results


def invert_dict(l):
    """
    Convert from list of dicts to dict of lists, where the key is the joined "pdb_code" and "rep" keys of the inner dictionary
    """
    out_dict = {}
    keys = l[0].keys()
    for key in keys:
        out_dict[key] = [i[key] for i in l]
    return out_dict


def main() -> None:
    logger.info("Defining file paths")
    pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
    pdb_files = [i for i in pdb_files if ".ca.pdb" not in i.name]

    i_start = int(sys.argv[1])
    i_stop = int(sys.argv[2])
    # Plan 01-08 (full-dataset batched run): an optional 3rd CLI arg names this
    # invocation's output dataset uniquely (e.g. "batch0003"). Without this, every
    # per-batch invocation of this script under the mandatory streaming/batching
    # architecture (Plan 01-07's checkpoint) would write to the same
    # atlas_derivatives_v2_{i_start}_{i_stop} directory whenever two batches
    # happen to have the same size (i_start/i_stop are always local indices into
    # whatever's currently in RAW_DATA_DIR/atlas, i.e. just the current batch,
    # since prior batches' raw files are deleted before the next download) --
    # silently overwriting or crashing on Dataset.save_to_disk's
    # directory-already-exists check. Defaults to the old "{i_start}_{i_stop}"
    # naming for backward compatibility with a single, non-batched invocation.
    batch_label = sys.argv[3] if len(sys.argv) > 3 else f"{i_start}_{i_stop}"
    # Plan 01-08 rework (2026-09-06): this loop was fully serial, the actual
    # cause of SLURM job 10202's ~7-day projection with 15/16 allocated CPUs
    # idle. Defaults to the SLURM allocation (multiprocessing.Pool, real
    # separate processes -- not threads, which the GIL would prevent from
    # parallelizing this CPU-bound MDTraj/mdigest work).
    n_jobs = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    pdb_codes = [pdb_f.stem for pdb_f in pdb_files]
    pdb_reps = [(pdb_code, rep) for pdb_code in pdb_codes for rep in range(1, N_REPS + 1)][
        i_start:i_stop
    ]
    total_jobs = len(pdb_reps)

    # Compute in parallel (multiprocessing.Pool, "spawn" context -- real
    # separate processes, each independently loads its own trajectory, so no
    # shared-memory risk; n_jobs bounds concurrent resident trajectories, the
    # original memory concern).
    logger.info(f"Computing {total_jobs} reps across {n_jobs} worker processes")
    results = parallel_pool(
        pdb_reps, compute_batched_trajectory_derivatives, n_jobs=n_jobs, report_every=50
    )

    logger.info("Creating HuggingFace dataset")
    ds = Dataset.from_dict(invert_dict(results))
    logger.info(f"Dataset created with {len(ds)} samples")
    out_path = ATLAS_PROCESSED_DATA_DIR / f"atlas_derivatives_v2_{batch_label}"
    logger.info(f"Saving dataset to {out_path}")
    if out_path.exists():
        # Re-running the same batch label (e.g. after a crash) should overwrite
        # its own prior (possibly partial) output, not crash on
        # save_to_disk's directory-already-exists check.
        import shutil

        shutil.rmtree(out_path)
    ds.save_to_disk(str(out_path))

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
    logger.info(
        f"Batch {batch_label} derivatives saved to {out_path} "
        "(cross-batch join happens once, after all batches, in run_full_atlas_pipeline.py)"
    )


if __name__ == "__main__":
    # Required: parallel_pool() now uses the "spawn" multiprocessing context
    # (see planet_md/parallel.py), which re-imports this file as a plain
    # module in every worker process. Everything above this guard is safe to
    # re-run (pure definitions); everything inside main() -- argv parsing,
    # globbing, and the actual Pool dispatch -- must NOT re-run in workers,
    # which is exactly what this guard prevents.
    main()
