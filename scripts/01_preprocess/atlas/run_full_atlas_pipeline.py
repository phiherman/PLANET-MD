"""
Full-scale ATLAS acquisition + processing pipeline (Plan 01-08).

Scales Plan 01-04's one-protein tracer to the full ATLAS dataset published at
dsimb.inserm.fr, under the MANDATORY streaming architecture decided by Plan
01-07's storage-capacity checkpoint (2026-09-06): download a bounded batch of
raw trajectories, run the full per-protein processing chain against that
batch, confirm the batch's outputs landed in `atlas_processed.h5`, THEN delete
that batch's raw `.pdb`/`.xtc` files before downloading the next batch. Never
hold more than one batch's raw trajectories on disk at once.

Resumable by design: batch completion is tracked in
`PROCESSED_DATA_DIR/atlas/atlas_pipeline_state.json`. Re-running this script
skips already-completed batches; the embeddings/3Di sub-scripts are
independently idempotent (skip-if-exists). Safe to re-run after a crash or
after a `checkpoint:human-verify` pause caused by a low-disk-space stop.

Usage (on staging-server, inside the planet_md conda env):
    python scripts/01_preprocess/atlas/run_full_atlas_pipeline.py \
        [--batch-size 300] [--min-free-gb 500] [--download-workers 8] \
        [--max-batches N]  # for smoke-testing a bounded number of batches

Designed to be launched as a long-running background/SLURM job (see
submit_atlas_process.sbatch) -- a full run over ATLAS's current ~1,938
published chains is a multi-day job dominated by network transfer from
dsimb.inserm.fr (~4-5 MB/s observed per connection) and per-protein GPU/CPU
compute (ESM3 embeddings, FoldSeek 3Di, MDTraj RMSF/CA-dist/autocorr, mdigest
GCC-LMI) -- not something that completes synchronously inside one shell
session.
"""

import argparse
import io
import json
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import Dataset, concatenate_datasets
from loguru import logger

from planet_md import config

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
ATLAS_H5 = ATLAS_PROCESSED_DATA_DIR / "atlas_processed.h5"
STATE_FILE = ATLAS_PROCESSED_DATA_DIR / "atlas_pipeline_state.json"
PDB_LIST_CACHE = ATLAS_PROCESSED_DATA_DIR / "atlas_pdb_list.txt"
API_BASE = "https://www.dsimb.inserm.fr/ATLAS/api"
SCRIPTS_DIR = Path(__file__).resolve().parent
N_REPS = 3


def get_free_gb(path: str = "/gpfs") -> float:
    _total, _used, free = shutil.disk_usage(path)
    return free / (1024**3)


def fetch_pdb_list() -> list[str]:
    """Fetch the full list of ATLAS PDB chain codes from the dsimb.inserm.fr
    'parsable' bulk-list endpoint (the same source `download_ATLAS.py` uses),
    caching it locally so repeated runs don't re-fetch."""
    if PDB_LIST_CACHE.exists():
        chains = [
            line.strip() for line in PDB_LIST_CACHE.read_text().splitlines() if line.strip()
        ]
        logger.info(f"Loaded {len(chains)} cached ATLAS chain IDs from {PDB_LIST_CACHE}")
        return chains

    logger.info(f"Fetching ATLAS chain list from {API_BASE}/parsable")
    r = requests.get(f"{API_BASE}/parsable", timeout=180)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    candidates = [n for n in z.namelist() if n.endswith("_ATLAS_pdb.txt")]
    if not candidates:
        raise RuntimeError("No *_ATLAS_pdb.txt found in parsable archive")
    with z.open(candidates[0]) as f:
        chains = sorted({line.decode().strip() for line in f if line.decode().strip()})

    ATLAS_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDB_LIST_CACHE.write_text("\n".join(chains) + "\n")
    logger.info(f"Fetched and cached {len(chains)} ATLAS chain IDs -> {PDB_LIST_CACHE}")
    return chains


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "completed_batches": [],
        "download_attempted": 0,
        "download_succeeded": 0,
        "download_failed_chains": [],
        "batch_footprints_du_sh": {},
    }


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def download_chain(chain: str) -> tuple[str, bool, str]:
    """Download one ATLAS chain's 'protein' archive (pdb + 3x xtc) via the
    per-chain REST endpoint (verified flat-zip layout: {chain}.pdb,
    {chain}_prod_R{1,2,3}_fit.xtc, {chain}_prod_R{1,2,3}.tpr, README.txt --
    matches Assumption A1 / Plan 01-04's confirmed naming convention exactly)."""
    out_dir = ATLAS_DATA_DIR / chain[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_f = out_dir / f"{chain}.pdb"
    xtc_ok = all(
        (out_dir / f"{chain}_prod_R{rep}_fit.xtc").exists() for rep in range(1, N_REPS + 1)
    )
    if pdb_f.exists() and xtc_ok:
        return chain, True, "already present"
    url = f"{API_BASE}/ATLAS/protein/{chain}"
    try:
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(out_dir)
        ok = pdb_f.exists() and all(
            (out_dir / f"{chain}_prod_R{rep}_fit.xtc").exists() for rep in range(1, N_REPS + 1)
        )
        if not ok:
            return chain, False, "zip extracted but expected files missing"
        return chain, True, "downloaded"
    except Exception as e:  # noqa: BLE001 -- must not abort the batch on one chain's failure
        return chain, False, str(e)


def download_batch(chains: list[str], workers: int) -> tuple[list[str], dict[str, str]]:
    succeeded, failures = [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_chain, c): c for c in chains}
        for fut in as_completed(futures):
            chain, ok, msg = fut.result()
            if ok:
                succeeded.append(chain)
            else:
                failures[chain] = msg
                logger.warning(f"Download failed for {chain}: {msg}")
    logger.info(
        f"Batch download: {len(succeeded)}/{len(chains)} succeeded, {len(failures)} failed"
    )
    return succeeded, failures


def run_step(cmd: list[str]) -> None:
    logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(config.PROJ_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Step failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")


def verify_batch_in_h5(chains: list[str]) -> list[str]:
    """Return the subset of `chains` that are missing required keys in
    atlas_processed.h5 after this batch's processing -- logged, not hidden."""
    import h5py

    missing = []
    if not ATLAS_H5.exists():
        return list(chains)
    with h5py.File(ATLAS_H5, "r") as f:
        for chain in chains:
            if chain not in f:
                missing.append(chain)
                continue
            grp = f[chain]
            if "embedding" not in grp or "struct_embedding" not in grp:
                missing.append(chain)
                continue
            ok = True
            for rep in range(1, N_REPS + 1):
                rep_key = f"R{rep}"
                if rep_key not in grp:
                    ok = False
                    break
                for label in ("rmsf", "ca_dist", "autocorr", "gcc_lmi", "shp"):
                    if label not in grp[rep_key]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                missing.append(chain)
    return missing


def delete_batch_raw(chains: list[str]) -> None:
    for chain in chains:
        out_dir = ATLAS_DATA_DIR / chain[:2]
        pdb_f = out_dir / f"{chain}.pdb"
        if pdb_f.exists():
            pdb_f.unlink()
        for rep in range(1, N_REPS + 1):
            for suffix in (f"_prod_R{rep}_fit.xtc", f"_prod_R{rep}.tpr"):
                fp = out_dir / f"{chain}{suffix}"
                if fp.exists():
                    fp.unlink()
        readme = out_dir / "README.txt"
        if readme.exists():
            readme.unlink()
        # remove the {chain[:2]} directory if now empty (other chains may share it)
        try:
            if out_dir.exists() and not any(out_dir.iterdir()):
                out_dir.rmdir()
        except OSError:
            pass


def process_batch(batch_idx: int, chains: list[str], state: dict, args: argparse.Namespace) -> None:
    batch_label = f"batch{batch_idx:04d}"
    logger.info(f"=== {batch_label}: {len(chains)} chains ===")

    free_gb = get_free_gb()
    logger.info(f"Free space before batch: {free_gb:.1f} GB")
    if free_gb < args.min_free_gb:
        raise SystemExit(
            f"STOP: free space {free_gb:.1f} GB below --min-free-gb {args.min_free_gb} GB "
            f"before starting {batch_label} -- refusing to download more raw data."
        )

    succeeded, failures = download_batch(chains, args.download_workers)
    state["download_attempted"] += len(chains)
    state["download_succeeded"] += len(succeeded)
    state["download_failed_chains"].extend(sorted(failures.keys()))
    save_state(state)

    if not succeeded:
        logger.warning(f"{batch_label}: nothing downloaded successfully, skipping processing")
        state["completed_batches"].append(batch_label)
        save_state(state)
        return

    du_before = subprocess.run(
        ["du", "-sh", str(ATLAS_DATA_DIR)], capture_output=True, text=True
    ).stdout.strip()
    logger.info(f"{batch_label} raw footprint after download: {du_before}")
    state["batch_footprints_du_sh"][batch_label] = du_before
    save_state(state)

    # Sequence + structure embeddings + 3Di (glob the whole RAW_DATA_DIR/atlas,
    # but since prior batches' raw was deleted, only this batch's proteins are
    # present -- all three scripts skip-if-output-exists, so re-running across
    # batches is safe and idempotent).
    run_step([sys.executable, str(SCRIPTS_DIR / "compute_sequence_embeddings.py")])
    run_step([sys.executable, str(SCRIPTS_DIR / "compute_structure_embeddings.py")])
    run_step([sys.executable, str(SCRIPTS_DIR / "compute_3di_sequences.py")])

    # RMSF / CA-dist / autocorr (CPU, needs raw .xtc present) -- writes
    # atlas_derivatives_v2_{batch_label} (unique per batch; process_atlas_cpu.py
    # was modified by this plan to accept a batch label instead of always
    # writing/joining into a single shared "atlas_derivatives_v2" dir, which
    # would collide across batches -- see Plan 01-08 SUMMARY deviations).
    n_reps_present = len(succeeded) * N_REPS
    run_step(
        [
            sys.executable,
            str(SCRIPTS_DIR / "process_atlas_cpu.py"),
            "0",
            str(n_reps_present),
            batch_label,
        ]
    )

    # H5 assembly for this batch (embeddings/SHP/basic derivatives) -- writes
    # into the persistent, cumulative atlas_processed.h5.
    run_step(
        [
            sys.executable,
            str(SCRIPTS_DIR / "build_atlas_h5.py"),
            f"atlas_derivatives_v2_{batch_label}",
        ]
    )

    # GCC-LMI (mdigest, needs raw .xtc present) -- globs whatever's currently
    # in RAW_DATA_DIR/atlas, i.e. exactly this batch; unmodified from Plan 01-04.
    run_step([sys.executable, str(SCRIPTS_DIR / "add_gcc_lmi_atlas.py")])

    missing = verify_batch_in_h5(succeeded)
    if missing:
        logger.warning(
            f"{batch_label}: {len(missing)}/{len(succeeded)} downloaded chains missing "
            f"complete H5 entries after processing: {missing}"
        )
        state.setdefault("incomplete_h5_chains", []).extend(missing)
    else:
        logger.info(f"{batch_label}: all {len(succeeded)} downloaded chains fully in H5")

    delete_batch_raw(succeeded)
    du_after = subprocess.run(
        ["du", "-sh", str(ATLAS_DATA_DIR)], capture_output=True, text=True
    ).stdout.strip()
    logger.info(f"{batch_label} raw footprint after delete: {du_after}")

    state["completed_batches"].append(batch_label)
    save_state(state)

    free_gb_after = get_free_gb()
    logger.info(f"Free space after {batch_label}: {free_gb_after:.1f} GB")
    if free_gb_after < args.min_free_gb:
        raise SystemExit(
            f"STOP: free space {free_gb_after:.1f} GB below --min-free-gb {args.min_free_gb} GB "
            f"after {batch_label} -- halting before next batch (checkpoint required)."
        )


def join_all_batches() -> None:
    """Concatenate every per-batch atlas_derivatives_v2_batchNNNN dataset into
    the final atlas_derivatives_v2, per Task 2 step 2's completeness check."""
    batch_dirs = sorted(ATLAS_PROCESSED_DATA_DIR.glob("atlas_derivatives_v2_batch*"))
    if not batch_dirs:
        logger.warning("No per-batch derivatives datasets found to join")
        return
    all_ds = [Dataset.load_from_disk(str(d)) for d in batch_dirs]
    joined = concatenate_datasets(all_ds)
    final_path = ATLAS_PROCESSED_DATA_DIR / "atlas_derivatives_v2"
    if final_path.exists():
        shutil.rmtree(final_path)
    joined.save_to_disk(str(final_path))
    logger.info(f"Joined {len(batch_dirs)} batch datasets -> {len(joined)} total rows -> {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--min-free-gb", type=float, default=500.0)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Process at most N new batches this invocation, then stop (for smoke-testing).",
    )
    args = parser.parse_args()

    if not ATLAS_H5.exists():
        ATLAS_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        import h5py

        h5py.File(ATLAS_H5, "w").close()
        logger.info(f"Bootstrapped empty {ATLAS_H5}")

    chains = fetch_pdb_list()
    logger.info(f"Full ATLAS chain list: {len(chains)} chains")

    batches = [chains[i : i + args.batch_size] for i in range(0, len(chains), args.batch_size)]
    state = load_state()
    done = set(state["completed_batches"])

    n_processed_this_run = 0
    for idx, batch in enumerate(batches):
        batch_label = f"batch{idx:04d}"
        if batch_label in done:
            continue
        if args.max_batches is not None and n_processed_this_run >= args.max_batches:
            logger.info(f"Reached --max-batches {args.max_batches}, stopping this invocation")
            break
        process_batch(idx, batch, state, args)
        n_processed_this_run += 1

    join_all_batches()

    logger.info(
        f"Run summary: {state['download_succeeded']}/{state['download_attempted']} "
        f"chain downloads succeeded across {len(state['completed_batches'])}/{len(batches)} batches"
    )
    if state["download_failed_chains"]:
        logger.warning(
            f"{len(state['download_failed_chains'])} chains failed download: "
            f"{state['download_failed_chains']}"
        )


if __name__ == "__main__":
    main()
