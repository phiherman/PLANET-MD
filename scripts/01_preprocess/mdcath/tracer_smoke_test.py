# THROWAWAY SCRIPT (Plan 01-05, Phase 01-checkpoint-validation).
#
# Not meant to be maintained -- Plan 01-06 supersedes this with the real,
# fixed scripts/03_evaluate/mdcath_validation.py. This script exists solely
# to prove the full mdCATH pipeline (raw per-domain download -> conversion ->
# ESM3 sequence/structure embeddings -> RMSF/CA-distance/autocorrelation ->
# GCC-LMI -> FoldSeek-3Di SHP -> H5 assembly -> v1 checkpoint load ->
# inference -> all four metrics) runs end to end on exactly one real mdCATH
# domain at the paper's own reference condition (320K, replica 0), before
# Plan 01-09 commits to the full ~5,184-domain / 3.3+ TB dataset.
import os

# NOTE: `planet_md.config` must be imported before `huggingface_hub` is
# imported anywhere in this process. `config`'s module-level `load_dotenv()`
# call sets HF_HOME/HUGGINGFACE_HUB_CACHE from .env; huggingface_hub reads
# those into frozen module-level constants the *first* time it is imported,
# and never re-reads them afterward. Getting this import order backwards
# causes a silent "never $HOME" constraint violation (see 01-04-SUMMARY.md
# for the empirical finding this pattern guards against).
from planet_md import config  # noqa: E402, F401

import torch  # noqa: E402
from loguru import logger  # noqa: E402

from planet_md.data.mdcath import (  # noqa: E402
    MDCATH_FOLDSEEK_CLUSTERS_FILE,
    MDCathDataModule,
)
from planet_md.metrics import (  # noqa: E402
    graph_diffusion_distance,
    ipsen_mikhailov_distance,
    mse,
)
from planet_md.modeling.architectures import PlanetMDModel  # noqa: E402

DOMAIN_ID = "12asA00"
TARGET_KEY = f"{DOMAIN_ID}/T320/R0"

# MDCathDataset._get_keys() reads keys directly from the opened H5 file
# itself (list(self._handle.keys())), unlike ATLASDataset which hardcodes a
# separate FoldSeek-clusters-file path for _get_keys() -- so no shim is
# needed there. MDCathDataModule.setup() (inherited from MDDataModule),
# however, still reads `clusters_file` (the module-level
# MDCATH_FOLDSEEK_CLUSTERS_FILE default) directly to compute the train/val/
# test split, so that file must still exist with our one domain's entry.
os.makedirs(MDCATH_FOLDSEEK_CLUSTERS_FILE.parent, exist_ok=True)
with open(MDCATH_FOLDSEEK_CLUSTERS_FILE, "w") as f:
    f.write(f"cluster_0\t{DOMAIN_ID}\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

logger.info("Loading v1 checkpoint from HF Hub (samsl/PLANET-MD)...")
model = PlanetMDModel.load_from_checkpoint(
    "v1", strict=True, HF_TOKEN=os.environ.get("HF_TOKEN")
)
model = model.to(device)
model.eval()

logger.info("Building MDCathDataModule against the tracer H5...")
mdl = MDCathDataModule(
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
mdl.setup("test")
logger.info(
    f"train/val/test sample counts: "
    f"{len(mdl.train_data)}/{len(mdl.val_data)}/{len(mdl.test_data)} "
    "(with only 1 FoldSeek cluster, int(train_pct * 1) == 0, so all 25 "
    "T{temp}/R{rep} combinations land in test_data by construction -- "
    "expected for this tracer)"
)

# Only 1 of the 25 T{temp}/R{rep} combinations for this domain has real
# rmsf/ca_dist/autocorr/shp labels populated (T320/R0, per Task 1's explicit
# I_START=0/I_STOP=1 scoping and Task 2's single-combination SHP/GCC-LMI
# run) -- gcc_lmi alone was computed for all 25 (add_gcc_lmi_mdcath.py's own
# unmodified full rep/temp loop). Index directly into the dataset for the
# one fully-populated sample rather than iterating test_data, which would
# KeyError on the other 24 incomplete samples.
target_idx = mdl.dataset.samples.index(TARGET_KEY)
logger.info(f"Targeting sample index {target_idx} ({TARGET_KEY})")


def run_inference(model, feats, device):
    model.eval()
    with torch.inference_mode():
        feats = {k: v.to(device).unsqueeze(0) for k, v in feats.items()}
        y_hat = model(feats)
    return {k: v.detach().cpu().squeeze() for k, v in y_hat.items()}


feats, labels = mdl.dataset[target_idx]
pred = run_inference(model, feats, device)

target_rmsf = labels["rmsf"].numpy()
pred_rmsf = pred["rmsf"].numpy()
logger.info(
    f"[{TARGET_KEY}] target['rmsf'].max() = {target_rmsf.max():.4f} -- raw "
    "units, unconverted MDTraj output (nm, not the paper's Angstroms; see "
    "RESEARCH.md Pitfall 7). Compare against Plan 01-04's ATLAS finding "
    "(1.01-1.25 across reps) to check whether both datasets' rmsf labels "
    "share the same unit."
)

rmsf_mse = mse(target_rmsf, pred_rmsf)
ca_dist_mse = mse(labels["ca_dist"].numpy(), pred["ca_dist"].numpy())
gcc_target = labels["gcc_lmi"].numpy()
gcc_pred = pred["gcc_lmi"].numpy()
gcc_imsd = ipsen_mikhailov_distance(gcc_target, gcc_pred)
gcc_gdd = graph_diffusion_distance(gcc_target, gcc_pred)
shp_mse = mse(labels["shp"].numpy(), pred["shp"].numpy())

print(f"[{TARGET_KEY}] rmsf_mse = {rmsf_mse:.6f}")
print(f"[{TARGET_KEY}] ca_dist_mse = {ca_dist_mse:.6f}")
print(f"[{TARGET_KEY}] gcc_lmi_ipsen_mikhailov_distance = {gcc_imsd:.6f}")
print(f"[{TARGET_KEY}] gcc_lmi_graph_diffusion_distance = {gcc_gdd:.6f}")
print(f"[{TARGET_KEY}] shp_mse = {shp_mse:.6f}")

logger.success(
    "Tracer smoke test complete: all four metrics computed end to end for "
    "one real mdCATH domain at the paper's reference condition (320K, R0)."
)
