# THROWAWAY SCRIPT (Plan 01-04, Phase 01-checkpoint-validation).
#
# Not meant to be maintained -- Plan 01-06 supersedes this with the real,
# fixed scripts/03_evaluate/atlas_validation.py. This script exists solely to
# prove the full ATLAS pipeline (raw download -> ESM3 sequence/structure
# embeddings -> RMSF/CA-distance/autocorrelation -> GCC-LMI -> FoldSeek-3Di
# SHP -> H5 assembly -> v1 checkpoint load -> inference -> all four metrics)
# runs end to end on exactly one real protein, before Plan 01-08 commits to
# the full ~1,389-protein ATLAS dataset.
import os

# NOTE: `planet_md.config` must be imported before `huggingface_hub` is
# imported anywhere in this process. `config`'s module-level `load_dotenv()`
# call sets HF_HOME/HUGGINGFACE_HUB_CACHE from .env; huggingface_hub reads
# those into frozen module-level constants (`huggingface_hub.constants.
# HF_HUB_CACHE`) the *first* time it is imported, and never re-reads them
# afterward. Getting this import order backwards causes a silent "never
# $HOME" constraint violation -- observed empirically this session (a stray
# ad hoc `hf_hub_download` call that imported `huggingface_hub` first landed
# a ~300MB checkpoint under `~/.cache/huggingface` instead of
# `/gpfs/projects/.../.cache/huggingface`; see 01-04-SUMMARY.md).
from planet_md import config  # noqa: E402, F401

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from planet_md.data.atlas import ATLAS_FOLDSEEK_CLUSTERS_FILE, ATLASDataModule  # noqa: E402
from planet_md.metrics import (  # noqa: E402
    graph_diffusion_distance,
    ipsen_mikhailov_distance,
    mse,
)
from planet_md.modeling.architectures import PlanetMDModel  # noqa: E402

PDB_CODE = "16pk_A"

# ATLASDataset._get_keys() reads this exact module-level path directly (not
# the `clusters_file` ATLASDataModule.__init__ accepts for its own train/
# val/test split computation) -- both must point at the same real file for
# this single-item tracer, so write our throwaway one-line, two-column,
# no-header TSV there (matches the format train_test_split_foldseek/
# ATLASDataset._get_keys() already parse: cluster_id, pdb_code).
os.makedirs(ATLAS_FOLDSEEK_CLUSTERS_FILE.parent, exist_ok=True)
with open(ATLAS_FOLDSEEK_CLUSTERS_FILE, "w") as f:
    f.write(f"cluster_0\t{PDB_CODE}\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

logger.info("Loading v1 checkpoint from HF Hub (samsl/PLANET-MD)...")
model = PlanetMDModel.load_from_checkpoint(
    "v1", strict=True, HF_TOKEN=os.environ.get("HF_TOKEN")
)
model = model.to(device)
model.eval()

logger.info("Building ATLASDataModule against the tracer H5...")
adl = ATLASDataModule(
    seq_features=True,
    struct_features=True,
    struct_stage="encoded",  # must match v1's own embedded hyper_parameters
    batch_size=1,
    shuffle=True,
    num_workers=0,
    train_pct=config.DEFAULT_PARAMETERS.train_pct,
    val_pct=config.DEFAULT_PARAMETERS.val_pct,
    random_seed=config.DEFAULT_PARAMETERS.random_seed,
    crop_size=config.DEFAULT_PARAMETERS.crop_size,
)
adl.setup("test")
logger.info(
    f"train/val/test sample counts: "
    f"{len(adl.train_data)}/{len(adl.val_data)}/{len(adl.test_data)} "
    "(with only 1 FoldSeek cluster, int(train_pct * 1) == 0, so all 3 "
    "reps land in test_data by construction -- expected for this tracer)"
)


def run_inference(model, feats, device):
    model.eval()
    with torch.inference_mode():
        feats = {k: v.to(device).unsqueeze(0) for k, v in feats.items()}
        y_hat = model(feats)
    return {k: v.detach().cpu().squeeze() for k, v in y_hat.items()}


for i, (feats, labels) in enumerate(adl.test_data):
    key = adl.dataset.samples[adl.test_data.indices[i]]
    pred = run_inference(model, feats, device)

    target_rmsf = labels["rmsf"].numpy()
    pred_rmsf = pred["rmsf"].numpy()
    logger.info(
        f"[{key}] target['rmsf'].max() = {target_rmsf.max():.4f} -- raw units, "
        "unconverted MDTraj output (nm, not the paper's Angstroms; see "
        "RESEARCH.md Pitfall 7 and 01-04-SUMMARY.md for the empirical finding)"
    )

    rmsf_mse = mse(target_rmsf, pred_rmsf)
    ca_dist_mse = mse(labels["ca_dist"].numpy(), pred["ca_dist"].numpy())
    gcc_target = labels["gcc_lmi"].numpy()
    gcc_pred = pred["gcc_lmi"].numpy()
    gcc_imsd = ipsen_mikhailov_distance(gcc_target, gcc_pred)
    gcc_gdd = graph_diffusion_distance(gcc_target, gcc_pred)
    shp_mse = mse(labels["shp"].numpy(), pred["shp"].numpy())

    print(f"[{key}] rmsf_mse = {rmsf_mse:.6f}")
    print(f"[{key}] ca_dist_mse = {ca_dist_mse:.6f}")
    print(f"[{key}] gcc_lmi_ipsen_mikhailov_distance = {gcc_imsd:.6f}")
    print(f"[{key}] gcc_lmi_graph_diffusion_distance = {gcc_gdd:.6f}")
    print(f"[{key}] shp_mse = {shp_mse:.6f}")

logger.success("Tracer smoke test complete: all four metrics computed end to end.")
