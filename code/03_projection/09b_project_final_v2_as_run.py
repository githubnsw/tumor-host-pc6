"""Project the FINAL v2 CPTAC expression (ex_RNA_Group v2, rule M) through the frozen AE with route B quantile mapping (FINAL method),
exactly as 07a but from the paper-format files, and write final_v2/inputs/ae_latent_pca10_external.csv (group = new group_num).
Also writes the raw-projection variant for reference. Run with mul-o (torch)."""
import os, sys, warnings
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); R1 = os.path.dirname(HERE); D = r"C:\Users\nsw\Desktop\body_composition"; V2 = os.path.join(R1, "final_v2", "inputs")
SCR = r"C:\Users\nsw\AppData\Local\Temp\claude\c--Users-nsw-Desktop-body-composition\61cfca76-a232-4ce7-a356-6d92f663b762\scratchpad"
sys.path.insert(0, SCR); from repro_pcs import Autoencoder, ALIAS_MAP, load_rna
os.chdir(D)
b = torch.load("ae_v3_outputs/ae_model_bundle.pt", map_location="cpu", weights_only=False)
genes = b["selected_genes"]; sm = np.asarray(b["scaler_mean"], np.float32); ss = np.asarray(b["scaler_scale"], np.float32); cfg = b["config"]
model = Autoencoder(cfg["input_dim"], cfg["hidden_dims"], cfg["latent_dim"], cfg.get("dropout", 0.3)); model.load_state_dict(b["state_dict"]); model.eval()
tc = load_rna([f"RNA_Group{i}_updated_to_new_symbol.csv" for i in range(1, 5)]); ref_sorted = {g: np.sort(tc.loc[g].values.astype(np.float32)) for g in genes}
NAMES = ["ex_RNA_Group1_LowMuscle_LowImmune.csv", "ex_RNA_Group2_LowMuscle_HighImmune.csv", "ex_RNA_Group3_HighMuscle_LowImmune.csv", "ex_RNA_Group4_HighMuscle_HighImmune.csv"]
ex = load_rna([os.path.join(V2, f) for f in NAMES]); gs = set(ex.index); n = ex.shape[1]
X_raw = np.zeros((n, len(genes)), np.float32); X_qn = np.tile(sm, (n, 1)).astype(np.float32); miss = []
for j, g in enumerate(genes):
    s = g if g in gs else (ALIAS_MAP.get(g) if ALIAS_MAP.get(g) in gs else None)
    if s is None: miss.append(g); continue
    v = ex.loc[s].values.astype(np.float32); X_raw[:, j] = v; r = pd.Series(v).rank(method="average").values; X_qn[:, j] = np.quantile(ref_sorted[g], (r - 0.5) / len(r))
print("v2 ex matrix", ex.shape, "| AE genes missing:", len(miss), miss[:10])
e2 = pd.read_csv(os.path.join(V2, "External_with_genes.csv")).set_index("patient_id")
for tag, X in [("raw", X_raw), ("B", X_qn)]:
    Z = (X - sm) / (ss + 1e-8)
    with torch.no_grad(): z = model.encoder(torch.FloatTensor(Z)).numpy()
    P = (z - np.asarray(b["pca10_mean"])) @ np.asarray(b["pca10_components"]).T
    out = pd.DataFrame(P, columns=[f"AE_PC{i+1}" for i in range(10)]); out.insert(0, "sample_id", ex.columns); out.insert(1, "patient_id", ex.columns); out.insert(2, "group", e2.loc[ex.columns, "group_num"].values)
    out.to_csv(os.path.join(V2, "ae_latent_pca10_external.csv" if tag == "B" else "ae_latent_pca10_external_rawprojection.csv"), index=False)
    old = pd.read_csv(os.path.join(HERE, "ruleM", f"ae_dir_{tag}", "ae_latent_pca10_external.csv")).set_index("patient_id").loc[ex.columns]
    print(tag, "PC6 Spearman vs 07a ruleM (59k-symbol matrix):", round(spearmanr(out.AE_PC6, old.AE_PC6)[0], 4), "| max |PC6 diff|", round(float((out.AE_PC6.values - old.AE_PC6.values).__abs__().max()), 4))
