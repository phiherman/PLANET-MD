import multiprocessing
from itertools import count

from loguru import logger
from tqdm import tqdm

# Plan 01-08 rework (2026-09-06): the default "fork" start method duplicates
# the calling process's live threads (e.g. numpy/OpenBLAS/OMP internal thread
# pools) into every worker, which COULD in principle deadlock a child that
# inherits a lock already held by a thread that no longer exists post-fork.
# This was originally (wrongly) believed to be why SLURM job 10203 froze
# (every worker showing wchan=futex_wait_queue, zero CPU progress). It
# wasn't: job 10204, run under this exact "spawn" context (which shares no
# thread/lock state with the parent at all), hit the IDENTICAL symptom --
# the real cause was memory exhaustion / swap thrashing under full
# concurrency (see scripts/01_preprocess/atlas/process_atlas_cpu.py and
# add_gcc_lmi_atlas.py for the actual incident and fix: a per-item memory
# footprint too large to run many-at-once, not a multiprocessing bug).
# "spawn" is kept anyway -- it remains strictly safer than "fork" for a
# process with live background threads, it just wasn't THE bug here. Callers
# using this function from a top-level script MUST guard that script's entry
# point with `if __name__ == "__main__":` -- spawn re-imports the calling
# module in each worker, so top-level side effects (argv parsing,
# re-dispatching work) must not live outside that guard.
_SPAWN_CTX = multiprocessing.get_context("spawn")


def create_batches(dataset, batch_size):
    """Create batches from a list of IDs."""
    dataset_size = len(dataset)
    for i in range(0, dataset_size, batch_size):
        yield [dataset[i] for i in range(i, min(i + batch_size, dataset_size))]


def process_batch_wrapper(args):
    """Wrapper to handle job numbering and batch processing."""
    job_num, batch, process_fn, report_every = args

    if report_every:
        logger.info(f"Job {job_num}: Processing batch of size {len(batch)}")

    try:
        result = process_fn(batch)

        if report_every and job_num % report_every == 0:
            logger.info(f"Job {job_num}: Completed successfully")

        return result

    except Exception as e:
        logger.info(f"Job {job_num}: Failed with error: {str(e)}")
        return []


def parallel_pool(dataset, process_fn, n_jobs=4, batch_size=32, report_every=None, maxtasksperchild=20):
    """
    Process an h5 file in parallel using a process pool.

    Args:
        dataset: List of IDs
        process_fn: Function that takes a batch of IDs and returns a list of results
        n_jobs: Number of parallel jobs to run
        batch_size: Size of batches to process
        report_every: How often to report progress (None for no reporting)
        maxtasksperchild: Recycle each worker after this many tasks (Plan
            01-08 rework, 2026-09-07 -- confirmed live that per-task memory in
            MDTraj/mdigest/MDAnalysis compute paths isn't always fully
            released within a long-lived worker; periodic recycling bounds
            the accumulation instead of letting it grow for the whole run)

    Returns:
        List of results from all batches, flattened
    """
    # total_batches = math.ceil(len(dataset) / batch_size)
    # batches = create_batches(dataset, batch_size)

    # # Create tuples of (job_num, batch, process_fn, report_every)
    # job_args = ((i, batch, process_fn, report_every)
    #             for i, batch in zip(count(), batches))

    # with Pool(processes=n_jobs) as pool:
    #     results = []
    #     for batch in tqdm(
    #         pool.imap(process_batch_wrapper, job_args),
    #         total=total_batches,
    #         desc="Processing batches"
    #     ):
    #         results.append(batch)

    # # Flatten results
    # return list(chain.from_iterable(results))

    total_items = len(dataset)
    job_args = (
        (i, [item], process_fn, report_every) for i, item in zip(count(), dataset)
    )

    with _SPAWN_CTX.Pool(processes=n_jobs, maxtasksperchild=maxtasksperchild) as pool:
        results = []
        for batch_results in tqdm(
            pool.imap(process_batch_wrapper, job_args),
            total=total_items,
            desc="Processing items",
        ):
            for r in batch_results:
                results.append(r)

    return results
