"""Hallmark-50 ssGSEA (same implementation as seed_check/hallmark_both_cohorts.py: Barbie 2009, alpha 0.25, min-max normalised) for the
FINAL v2 CPTAC expression (rule M). Output: final_v2/hallmark_ssgsea_CPTAC_tpm_v2.csv (same layout as seed_check/hallmark_ssgsea_CPTAC_tpm.csv)."""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); V2 = os.path.join(HERE, "inputs"); D = r"C:\Users\nsw\Desktop\body_composition"
sets = {}
for line in open(os.path.join(D, "seed_check", "MSigDB_Hallmark_2020.gmt"), encoding="utf-8"):
    p = line.rstrip("\n").split("\t"); sets[p[0].upper().replace(" ", "_").replace("-", "_")] = [g for g in p[2:] if g]


def load_rna(files):
    dfs = [pd.read_csv(f, index_col=0) for f in files]; dfs = [d[~d.index.duplicated()] for d in dfs]
    return np.log2(pd.concat(dfs, axis=1, join="inner").fillna(0) + 1)


def ssgsea(X, sets, alpha=0.25):
    genes = X.index.values; n = len(genes); ranks = X.rank(axis=0, method="average").values; out = {}; order = np.argsort(-ranks, axis=0)
    for name, gl in sets.items():
        mask = pd.Index(genes).isin(gl)
        if mask.sum() < 5: continue
        es = np.empty(X.shape[1])
        for j in range(X.shape[1]):
            o = order[:, j]; inset = mask[o]; r = ranks[o, j]; w = np.where(inset, r ** alpha, 0.0); P = np.cumsum(w) / w.sum(); Q = np.cumsum(~inset) / (n - inset.sum()); es[j] = (P - Q).sum()
        out[name] = es
    S = pd.DataFrame(out, index=X.columns); return (S - S.values.min()) / (S.values.max() - S.values.min())


X = load_rna([os.path.join(V2, f) for f in ["ex_RNA_Group1_LowMuscle_LowImmune.csv", "ex_RNA_Group2_LowMuscle_HighImmune.csv", "ex_RNA_Group3_HighMuscle_LowImmune.csv", "ex_RNA_Group4_HighMuscle_HighImmune.csv"]])
S = ssgsea(X, sets); S.to_csv(os.path.join(HERE, "hallmark_ssgsea_CPTAC_tpm_v2.csv")); print("done", S.shape)
old = pd.read_csv(os.path.join(D, "seed_check", "hallmark_ssgsea_CPTAC_tpm.csv"), index_col=0).loc[S.index]
print("Spearman v1 vs v2 per set (median):", round(float(np.median([S[c].corr(old[c], method="spearman") for c in S.columns if c in old.columns])), 3))
