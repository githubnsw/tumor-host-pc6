"""Project the CPTAC-3 expression matrix through the frozen TCGA autoencoder + PCA (route B: per-gene quantile mapping to the
TCGA reference distribution, then the fixed scaler/encoder/PCA). Reproduces data/ae_latent_pca10_external.csv.
Usage (needs torch):  python code/03_projection/project_cptac.py"""
import os, sys, gzip
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(REPO, "code"))
from ae_model import ALIAS_MAP, load_bundle, encode_to_pcs
bundle, model = load_bundle(os.path.join(REPO, "model", "ae_v3_outputs", "ae_model_bundle.pt")); genes = list(bundle["selected_genes"])
ref = np.load(os.path.join(REPO, "data", "expression", "tcga_quantile_reference_3000genes.npz")); ref_sorted = dict(zip(ref["genes"].tolist(), ref["sorted_log2"]))
with gzip.open(os.path.join(REPO, "data", "expression", "cptac_expr_tpm_179.csv.gz"), "rt") as g: ex = pd.read_csv(g, index_col=0)
ex = np.log2(ex[~ex.index.duplicated(keep="first")].fillna(0) + 1); gs = set(ex.index); n = ex.shape[1]
sm = np.asarray(bundle["scaler_mean"], np.float32); X_raw = np.zeros((n, len(genes)), np.float32); X_qn = np.tile(sm, (n, 1)).astype(np.float32); missing = []
for j, g in enumerate(genes):
    s = g if g in gs else (ALIAS_MAP.get(g) if ALIAS_MAP.get(g) in gs else None)
    if s is None: missing.append(g); continue                      # absent in CPTAC: kept at the training mean (-> 0 after scaling)
    v = ex.loc[s].values.astype(np.float32); X_raw[:, j] = v
    r = pd.Series(v).rank(method="average").values; X_qn[:, j] = np.quantile(ref_sorted[g], (r - 0.5) / len(r))
print(f"CPTAC samples {n}; AE genes missing in CPTAC: {len(missing)}")
out_dir = os.path.join(REPO, "results", "reproduced"); os.makedirs(out_dir, exist_ok=True)
for tag, X in [("quantile_mapped", X_qn), ("raw_projection", X_raw)]:
    P, _ = encode_to_pcs(bundle, model, X); P.insert(0, "patient_id", ex.columns); P.to_csv(os.path.join(out_dir, f"ae_latent_pca10_external__{tag}.csv"), index=False)
ref_pc = pd.read_csv(os.path.join(REPO, "data", "ae_latent_pca10_external.csv")).set_index("patient_id")
new = pd.read_csv(os.path.join(out_dir, "ae_latent_pca10_external__quantile_mapped.csv")).set_index("patient_id").loc[ref_pc.index]
print("max |PC6 difference| vs data/ae_latent_pca10_external.csv:", float(np.abs(new.AE_PC6.values - ref_pc.AE_PC6.values).max()))
