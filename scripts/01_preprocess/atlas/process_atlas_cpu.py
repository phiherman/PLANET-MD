# %% # Load packages
import os

# Plan 01-08 rework (2026-09-06): must be set before numpy/mdtraj/BLAS are
# imported anywhere in this process. With 16 worker processes already
# parallelizing across the SLURM allocation's CPUs, each worker ALSO letting
# OpenBLAS/MKL/OMP spin up its own internal thread pool oversubscribes the
# machine (16 workers x N BLAS threads > allocated CPUs). NOTE (2026-09-07):
# this was originally written believing it also explained SLURM job 10203's
# apparent deadlock (fork()-inherited, already-locked BLAS thread) -- that
# theory was WRONG. Job 10204 hit the identical wchan=futex_wait_queue /
# zero-CPU-progress symptom AFTER switching to the "spawn" context (which
# does not inherit parent threads at all), proving fork-inherited locks were
# never the cause. The real cause was memory exhaustion / swap thrashing from
# compute_contacts' per-item footprint under full concurrency -- see main()'s
# LARGE_RESIDUE_THRESHOLD comment and planet_md.trajectory's
# compute_ca_dist_and_autocorrelation_from_compact() docstring for the actual
# incident and fix. Kept anyway: pinning BLAS/OMP to 1 thread each is still a
# real, independent improvement (avoids the oversubscription noted above).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import sys
from itertools import combinations

import mdtraj as md
import numpy as np
from datasets import Dataset
from loguru import logger

from planet_md import config
from planet_md.parallel import parallel_pool
from planet_md.trajectory import (
    compute_ca_dist_and_autocorrelation_from_compact,
    compute_rmsf,
    normalize,
)

# Plan 01-08 rework (2026-09-07): chains at/above this residue count use a
# separate, low-concurrency pass (see main()) -- see LARGE_RESIDUE_THRESHOLD's
# comment there for why.
LARGE_RESIDUE_THRESHOLD = 400
# 3 workers x this batch's worst case (1gte_D, 1025 residues, ~21GB compact
# form) = ~63GB peak -- verified batch0000 never exceeded 40GB at 2 workers,
# and the node has 251GB physical (SLURM's --mem isn't actually enforced on
# this cluster; 251GB physical, shared with other users, is the real
# ceiling). Bumped from 2 -> 3 (2026-09-07) after batch0000 completed cleanly
# at 2, trading some margin for ~33% less wall-clock on this phase across the
# remaining 6 batches. If a future batch's largest chain is bigger than
# 1gte_D, re-derive this bound rather than assuming it still holds.
LARGE_PROTEIN_WORKERS = 3

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

    # Memory-efficient contacts (Plan 01-08 rework, 2026-09-07): the compact
    # (non-squareform) form from md.compute_contacts is HALF the size of the
    # full (n_frames, n_res, n_res) squareform tensor compute_contacts()
    # would produce -- no symmetric duplication. See
    # compute_ca_dist_and_autocorrelation_from_compact()'s docstring for why
    # this matters (a real memory-exhaustion incident on staging-server).
    d, pairs = md.compute_contacts(
        traj,
        contacts=list(combinations(np.arange(traj.n_residues), 2)),
        scheme="ca",
        ignore_nonprotein=True,
    )
    ca_dist, autocorr = compute_ca_dist_and_autocorrelation_from_compact(
        d, pairs, traj.n_residues
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
    del traj, d, pairs, rmsf, ca_dist, autocorr
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

    # Plan 01-08 rework (2026-09-07): the per-item memory cost of
    # RMSF/CA-dist/autocorr scales as O(n_frames * n_residues^2) (a dense
    # per-residue-pair time series). Confirmed live on staging-server: with
    # ATLAS chains up to ~1000 residues (10001-frame trajectories), a SINGLE
    # such chain's compact contacts array is ~20GB; running the full 16-way
    # pool with several large chains landing concurrently drove combined
    # worker RSS to 208GB against a 64GB SLURM allocation, pushed the shared
    # node's swap to 99% full, and made the run LOOK hung (multi-hour zero
    # CPU progress) when it was really thrashing -- not a deadlock, not a
    # multiprocessing bug (this recurred identically under both "fork" and
    # "spawn" contexts, ruling that out). Large chains get a small dedicated
    # pool (bounded worst case: LARGE_PROTEIN_WORKERS * ~20GB, safely under
    # the 64GB allocation); everything else keeps full n_jobs concurrency
    # (worst case: n_jobs * ~5GB for a ~400-residue chain, also safely under
    # budget). The two passes run sequentially, so their peaks never stack.
    pdb_n_residues = {}
    for pdb_code in {c for c, _ in pdb_reps}:
        pdb_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}.pdb"
        pdb_n_residues[pdb_code] = md.load(str(pdb_f)).n_residues

    large_reps = [pr for pr in pdb_reps if pdb_n_residues[pr[0]] >= LARGE_RESIDUE_THRESHOLD]
    small_reps = [pr for pr in pdb_reps if pdb_n_residues[pr[0]] < LARGE_RESIDUE_THRESHOLD]

    results = []
    if large_reps:
        logger.info(
            f"{len(large_reps)}/{total_jobs} reps from chains >={LARGE_RESIDUE_THRESHOLD} "
            f"residues: processing with {LARGE_PROTEIN_WORKERS} workers (memory-bounded)"
        )
        results += parallel_pool(
            large_reps,
            compute_batched_trajectory_derivatives,
            n_jobs=min(LARGE_PROTEIN_WORKERS, n_jobs),
            report_every=10,
        )
    if small_reps:
        logger.info(f"{len(small_reps)}/{total_jobs} remaining reps across {n_jobs} worker processes")
        results += parallel_pool(
            small_reps, compute_batched_trajectory_derivatives, n_jobs=n_jobs, report_every=50
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
