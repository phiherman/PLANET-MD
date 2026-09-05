# %%
import h5py
import torch
from datasets import Dataset
from tqdm import tqdm

from planet_md import config
from planet_md.data.utils import update_h5_dataset
from planet_md.trajectory import convert_to_normalized_shp

MDCATH_DATA_DIR = config.RAW_DATA_DIR / "mdcath"
MDCATH_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "mdcath"
MDCATH_H5 = MDCATH_PROCESSED_DATA_DIR / "mdcath_processed.h5"

mdcath_derivatives_data = Dataset.load_from_disk(
    str(MDCATH_PROCESSED_DATA_DIR / "mdcath_derivatives_v2")
)
# NOTE (tracer, Plan 01-05): the pre-built `fs_shp/dataset` HuggingFace Dataset
# the original SHP block below assumed (built by build_shp_3di_disbatch.py,
# aggregating many disBatch-parallelized single-item .pt outputs from
# foldseek_shp_disbatch_single.py) does not exist for a from-scratch,
# single-domain tracer run -- and `disBatch` itself is NOT installed on
# staging-server (confirmed this session: `which disBatch` -> not found).
# This tracer instead reads each domain/rep/temp's single `.pt` file written
# directly by foldseek_shp_disbatch_single.py, applies the same
# convert_to_normalized_shp() normalization the aggregation step would have
# applied, and writes straight into the H5 -- skipping the intermediate
# Dataset round-trip entirely. Plan 01-09 must decide whether to keep this
# direct-to-H5 shortcut at full scale (e.g. via a plain loop or SLURM array
# job over domains, since disBatch is unavailable here) or install disBatch
# and reconstruct build_shp_3di_disbatch.py's intermediate Dataset step.

# %%
# Build file
with h5py.File(MDCATH_H5, "a") as h5file:
    # Basic Derivatives
    for replicate in tqdm(mdcath_derivatives_data, desc="Basic Derivatives"):
        pdb_id = replicate["pdb_code"]
        rep = replicate["rep"]
        temp = replicate["temp"]
        # logger.info(f"Autocorr: Processing {pdb_id} R{rep}")

        for key in ["rmsf", "ca_dist", "autocorr"]:
            update_h5_dataset(h5file, f"{pdb_id}/T{temp}/R{rep}/{key}", replicate[key])

    # Foldseek SHP (direct single-.pt-file shortcut, see NOTE above -- no
    # intermediate fs_shp/dataset round-trip for this tracer)
    for replicate in tqdm(mdcath_derivatives_data, desc="Foldseek SHP (direct .pt read)"):
        pdb_id = replicate["pdb_code"]
        rep = replicate["rep"]
        temp = replicate["temp"]
        shp_pt_file = (
            MDCATH_PROCESSED_DATA_DIR
            / "fs_shp"
            / pdb_id
            / f"{pdb_id}_rep_{rep}_temp{temp}.pt"
        )
        result = torch.load(shp_pt_file, weights_only=False)
        shp = convert_to_normalized_shp(result["fs_shp"], max_dim=20)
        # logger.info(f"SHP: Processing {pdb_id} R{rep}")
        update_h5_dataset(h5file, f"{pdb_id}/T{temp}/R{rep}/shp", shp, overwrite=True)

    # Sequence Embeddings
    for pdb_code in tqdm(
        list(set(mdcath_derivatives_data["pdb_code"])), desc="Sequence embeddings"
    ):
        seq_file = (
            MDCATH_PROCESSED_DATA_DIR / "seq_embeddings" / pdb_code / f"{pdb_code}.seq"
        )
        seq_embedding = torch.load(
            seq_file, map_location="cpu", weights_only=True
        ).squeeze()[1:-1]
        update_h5_dataset(h5file, f"{pdb_code}/embedding", seq_embedding, overwrite=True)

    # Structural Embeddings
    for pdb_code in tqdm(
        list(set(mdcath_derivatives_data["pdb_code"])), desc="Structural embeddings"
    ):
        struct_file = (
            MDCATH_PROCESSED_DATA_DIR
            / "struct_embeddings"
            / pdb_code
            / f"{pdb_code}.struct"
        )
        struct_embedding = torch.load(
            struct_file, map_location="cpu", weights_only=True
        )
        for k, v in struct_embedding.items():
            update_h5_dataset(
                h5file, f"{pdb_code}/struct_embedding/{k}", v.squeeze(), overwrite=True
            )
