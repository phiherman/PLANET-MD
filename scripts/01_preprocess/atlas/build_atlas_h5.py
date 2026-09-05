# %%
import h5py
import torch
from datasets import Dataset
from tqdm import tqdm

from planet_md import config
from planet_md.data.utils import update_h5_dataset
from planet_md.trajectory import FS_3DI_LIST

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"
ATLAS_H5 = ATLAS_PROCESSED_DATA_DIR / "atlas_processed.h5"
FS_3DI_DIR = ATLAS_PROCESSED_DATA_DIR / "3di"

atlas_derivatives_data = Dataset.load_from_disk(
    str(ATLAS_PROCESSED_DATA_DIR / "atlas_derivatives_v2")
)
# NOTE (tracer, Plan 01-04): the pre-built `esm_shp/dataset` and `fs_shp/dataset`
# HuggingFace Datasets the original SHP block below assumed do not exist for a
# from-scratch run (RESEARCH.md D-02 -- no preprocessed intermediates ship
# anywhere). This tracer reads the single `.3di` FoldSeek descriptor file
# directly instead (see the "SHP" block below). Plan 01-08 must decide whether
# to keep this direct-3di-to-H5 approach at full scale or reconstruct the
# intermediate `fs_shp`/`esm_shp` dataset-building step this script originally
# assumed existed.

# %%
# Build file
with h5py.File(ATLAS_PROCESSED_DATA_DIR / "atlas_processed.h5", "r+") as h5file:
    # Basic Derivatives
    for replicate in tqdm(atlas_derivatives_data, desc="Basic Derivatives"):
        pdb_id = replicate["pdb_code"]
        rep = replicate["rep"]
        # logger.info(f"Autocorr: Processing {pdb_id} R{rep}")

        for key in ["rmsf", "ca_dist", "autocorr"]:
            update_h5_dataset(h5file, f"{pdb_id}/R{rep}/{key}", replicate[key])

    # SHP
    def convert_to_normalized_shp(preshp, max_dim=len(FS_3DI_LIST)):
        preshp = torch.as_tensor(preshp)
        # Only squeeze away *extra* dims beyond 2 -- a genuine (num_frames=1,
        # seq_len) input (as produced by a single static-structure 3Di
        # descriptor, not a multi-frame trajectory ensemble) must keep its
        # frame axis, or plain .squeeze() collapses it to 1D and breaks the
        # transpose below (RuntimeError: bincount only supports 1-d input).
        if preshp.dim() > 2:
            preshp = preshp.squeeze()
        shp = torch.stack([torch.bincount(i, minlength=max_dim) for i in preshp.T])
        shp = shp.T / shp.sum(axis=1)
        return shp.T

    # Tracer (Plan 01-04): SHP is structure-only, so the same descriptor applies
    # to every rep of a given static structure -- read once per pdb_code, write
    # to every rep's `shp` key. `structureto3didescriptor`'s output format is
    # tab-separated: [0] name/header, [1] amino-acid sequence, [2] 3Di sequence
    # (index-encoded via FS_3DI_LIST below), [3] per-residue numeric descriptors
    # (unused here) -- format verified empirically this session, not documented
    # by foldseek's own --help.
    pdb_codes = sorted(set(atlas_derivatives_data["pdb_code"]))
    reps = sorted(set(atlas_derivatives_data["rep"]))
    for pdb_id in tqdm(pdb_codes, desc="SHP (from .3di descriptor)"):
        di_file = FS_3DI_DIR / pdb_id[:2] / f"{pdb_id}.3di"
        with open(di_file) as f:
            di_seq = f.readline().rstrip("\n").split("\t")[2]
        preshp = torch.tensor([[FS_3DI_LIST.index(c) for c in di_seq]])
        shp = convert_to_normalized_shp(preshp)
        for rep in reps:
            update_h5_dataset(h5file, f"{pdb_id}/R{rep}/shp", shp, overwrite=True)

    # Sequence Embeddings
    for pdb_code in tqdm(pdb_codes, desc="Sequence embeddings"):
        seq_file = ATLAS_PROCESSED_DATA_DIR / "seq_embeddings" / pdb_code[:2] / f"{pdb_code}.seq"
        seq_embedding = torch.load(seq_file, map_location="cpu", weights_only=True).squeeze()[1:-1]
        update_h5_dataset(h5file, f"{pdb_code}/embedding", seq_embedding, overwrite=True)

    # Structural Embeddings
    for pdb_code in tqdm(pdb_codes, desc="Structural embeddings"):
        struct_file = ATLAS_PROCESSED_DATA_DIR / "struct_embeddings" / pdb_code[:2] / f"{pdb_code}.struct"
        struct_embedding = torch.load(struct_file, map_location="cpu", weights_only=True)
        for k, v in struct_embedding.items():
            update_h5_dataset(h5file, f"{pdb_code}/struct_embedding/{k}", v.squeeze(), overwrite=True)

# %%
