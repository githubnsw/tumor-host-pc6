"""A. Prepare inputs for the cancer-type-adjusted DEG analysis (TCGA):
 - log2(RSEM+1) expression matrix (genes x patients) as used by the Fig 4 cell
 - group labels under both definitions: cohort-wide medians (submitted) and within-cancer-type medians (new)
Writes expr_log2_tcga.csv, groups_tcga.csv"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R1 = os.path.dirname(os.path.dirname(HERE))
FIN = os.path.join(R1, "final_rerun", "inputs"); D = r"C:\Users\nsw\Desktop\body_composition"
clin = pd.read_csv(os.path.join(FIN, "TCGA_with_genes.csv")); pcs = pd.read_csv(os.path.join(FIN, "ae_latent_pca10_all.csv"))
clin["pid"] = clin.patient_id.astype(str).str[:12]; pcs["pid"] = pcs.patient_id.astype(str).str[:12]
m = clin[["pid", "OS_time", "OS", "SMA_zscore", "cancer_type", "sex", "age", "stage"]].merge(pcs[["pid", "AE_PC6"]].drop_duplicates("pid"), on="pid")
m["PC6_z"] = (m.AE_PC6 - m.AE_PC6.mean()) / m.AE_PC6.std()
m["OS_months"] = m.OS_time / 30.44
m = m.dropna(subset=["PC6_z", "SMA_zscore", "OS_months", "OS"])   # same filter as the Fig 4 cell
for tag, within in [("cohortwide", False), ("withintype", True)]:
    pm = m.groupby("cancer_type").PC6_z.transform("median") if within else m.PC6_z.median()
    sm = m.groupby("cancer_type").SMA_zscore.transform("median") if within else m.SMA_zscore.median()
    m[f"group4_{tag}"] = np.where(m.PC6_z >= pm, "PC6-High", "PC6-Low") + " / " + np.where(m.SMA_zscore >= sm, "SMA-High", "SMA-Low")
    m[f"worst_{tag}"] = (m[f"group4_{tag}"] == "PC6-High / SMA-Low").astype(int)
dfs = [pd.read_csv(os.path.join(D, f"RNA_Group{i}_updated_to_new_symbol.csv")).set_index("gene") for i in range(1, 5)]
rna = pd.concat(dfs, axis=1); rna = rna.loc[~rna.index.duplicated()]; rna.columns = [c[:12] for c in rna.columns]
rna = rna.loc[:, ~rna.columns.duplicated()]
common = [p for p in m.pid if p in rna.columns]; m = m[m.pid.isin(common)]
rna_log = np.log2(rna[common].astype(float).fillna(0) + 1)   # NaN -> 0 as in the AE pipeline (autoe.py)
rna_log = rna_log[((rna_log > 0).sum(axis=1) >= 5) & (rna_log.var(axis=1) > 0)]   # drop near-empty / constant genes
rna_log.to_csv(os.path.join(HERE, "expr_log2_tcga.csv")); m.to_csv(os.path.join(HERE, "groups_tcga.csv"), index=False)
print("expr", rna_log.shape, "patients", len(m)); print(m.group4_withintype.value_counts().to_dict()); print(m.group4_cohortwide.value_counts().to_dict())
print(pd.crosstab(m.cancer_type, m.worst_withintype))
