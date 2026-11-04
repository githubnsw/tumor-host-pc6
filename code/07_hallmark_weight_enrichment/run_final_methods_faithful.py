"""PC6 Biological Characterization — FINAL, Methods-faithful run (approved 2026-11-03).
Step 1  reused: official Hallmark-50 weight-based enrichment (../tables/step1_*). No rerun.
Step 2  gene-level Spearman (PC6 z vs log2 expression) for ALL member genes of the 8 approved Hallmark sets
        (PC6 BH q<0.05 in Step 1, both directions); each gene tested once per cohort; BH over all unique genes per cohort;
        all genes reported. Plus the original 7-gene proliferation marker set (curated; reused from ../tables/step3_targeted_*).
Step 3  ssGSEA with the submitted implementation and settings (gseapy.ssgsea, sample_norm_method='rank', min_size=2, NES)
        for the same 8 Hallmark sets; Spearman vs PC6 z; BH over the 8 sets per cohort.
Inputs frozen (FINAL v2); no model/PCA/PC6/survival changes; nothing in submitted folders is modified.
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import gseapy as gp

HERE = os.path.dirname(os.path.abspath(__file__)); PAR = os.path.dirname(HERE); R1 = os.path.dirname(PAR); D = os.path.dirname(R1)
FIN = os.path.join(R1, "final_v2", "inputs"); OUT = os.path.join(HERE, "tables"); FIG = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
GMT = os.path.join(R1, "r2_immune_programs_sma", "sources", "h.all.v2023.2.Hs.symbols.gmt")
log = []
def say(s): print(s); log.append(s)
say(f"gseapy {gp.__version__}")

# ---- Step 1 (reused): approved pathway list = PC6 q<0.05, both directions ----
S1 = pd.read_csv(os.path.join(PAR, "tables", "step1_weight_enrichment_all_PCs_hallmark50.csv"))
P6 = S1[S1.PC == "PC6"].sort_values("p"); SEL = P6[P6.q < 0.05].copy()
SEL["step1_direction"] = np.where(SEL.mean_weight > 0, "positive", "negative")
SEL = SEL.sort_values("mean_weight", ascending=False)
SEL[["pathway", "n_overlap", "n_set", "mean_weight", "median_weight", "p", "q", "step1_direction"]].to_csv(os.path.join(OUT, "step1_PC6_approved_pathways.csv"), index=False)
APPROVED = SEL.pathway.tolist()
say("Step 1 (reused) — approved pathways (PC6 BH q<0.05, both directions): " + ", ".join(s.replace("HALLMARK_", "") + ("(+)" if d == "positive" else "(−)") for s, d in zip(SEL.pathway, SEL.step1_direction)))
assert len(APPROVED) == 8, len(APPROVED)

H = {}
for line in open(GMT, encoding="utf-8"):
    p = line.rstrip("\n").split("\t"); H[p[0]] = [g for g in p[2:] if g]
sh = lambda s: s.replace("HALLMARK_", "")

# ---- data (same loaders / standardisation as the submitted Fig 3 cell) ----
def load_expr(files, cut12):
    dfs = []
    for f in files:
        d = pd.read_csv(f); d = d.set_index(d.columns[0]); d = d[~d.index.duplicated()]; dfs.append(d)
    X = pd.concat(dfs, axis=1, join="inner"); X = X.loc[:, ~X.columns.duplicated()]
    if cut12: X.columns = [c[:12] for c in X.columns]; X = X.loc[:, ~X.columns.duplicated()]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return np.log2(X + 1)
def load_cohort(coh):
    if coh == "TCGA":
        X = load_expr([os.path.join(D, f"RNA_Group{i}_updated_to_new_symbol.csv") for i in range(1, 5)], True)
        pc = pd.read_csv(os.path.join(FIN, "ae_latent_pca10_all.csv")); pc["pid"] = pc["patient_id"].astype(str).str[:12]
    else:
        X = load_expr([os.path.join(FIN, f) for f in ["ex_RNA_Group1_LowMuscle_LowImmune.csv", "ex_RNA_Group2_LowMuscle_HighImmune.csv",
                                                     "ex_RNA_Group3_HighMuscle_LowImmune.csv", "ex_RNA_Group4_HighMuscle_HighImmune.csv"]], False)
        pc = pd.read_csv(os.path.join(FIN, "ae_latent_pca10_external.csv")); pc["pid"] = pc["patient_id"].astype(str)
    pc = pc.drop_duplicates("pid"); pc["PC6_z"] = (pc["AE_PC6"] - pc["AE_PC6"].mean()) / pc["AE_PC6"].std()
    common = [p for p in pc["pid"] if p in X.columns]
    X = X[common]; pc6 = pc.set_index("pid").loc[common, "PC6_z"].astype(float)
    say(f"[{coh}] {X.shape[0]} genes x {len(common)} patients")
    return X, pc6
DATA = {c: load_cohort(c) for c in ["TCGA", "CPTAC-3"]}

# ================= Step 2: gene-level Spearman, 8 approved pathways =================
say("\nStep 2 — gene-level Spearman (each gene tested once per cohort; BH over all unique tested genes per cohort)")
members = {s: H[s] for s in APPROVED}
union = sorted({g for s in APPROVED for g in H[s]})
G2, SUMM2 = [], []
for coh, (X, pc6) in DATA.items():
    rows = []
    for g in union:
        if g not in X.index: continue
        x = X.loc[g].values.astype(float)
        if np.nanstd(x) == 0: continue
        rho, p = spearmanr(pc6.values, x); rows.append(dict(cohort=coh, gene=g, rho=rho, p=p))
    gdf = pd.DataFrame(rows); gdf["q"] = multipletests(gdf["p"], method="fdr_bh")[1]
    gdf["pathways"] = gdf.gene.map(lambda g: ";".join(sh(s) for s in APPROVED if g in set(H[s])))
    gdf["n_pathways"] = gdf.pathways.str.count(";") + 1
    say(f"  [{coh}] unique genes tested {len(gdf)} (of {len(union)} in the 8 sets); q<0.05: {int((gdf.q < 0.05).sum())} (positive {int(((gdf.q < 0.05) & (gdf.rho > 0)).sum())}, negative {int(((gdf.q < 0.05) & (gdf.rho < 0)).sum())})")
    G2.append(gdf)
    for s in APPROVED:
        g = gdf[gdf.gene.isin(set(H[s]))].sort_values("rho", ascending=False)
        d1 = SEL.set_index("pathway").loc[s, "step1_direction"]; sgn = 1 if d1 == "positive" else -1
        same = g[np.sign(g.rho) == sgn]; k = max(1, int(np.ceil(0.1 * len(g))))
        pos = (g.rho * sgn).clip(lower=0); share = pos.sort_values(ascending=False).head(k).sum() / max(pos.sum(), 1e-9)
        top = g.reindex(g.rho.abs().sort_values(ascending=False).index).head(10)
        SUMM2.append(dict(cohort=coh, pathway=s, step1_direction=d1, n_set=len(H[s]), n_tested=len(g), mean_rho=g.rho.mean(), median_rho=g.rho.median(),
                          IQR_lo=g.rho.quantile(.25), IQR_hi=g.rho.quantile(.75), n_pos=int((g.rho > 0).sum()), n_neg=int((g.rho < 0).sum()),
                          n_q05_pos=int(((g.q < 0.05) & (g.rho > 0)).sum()), n_q05_neg=int(((g.q < 0.05) & (g.rho < 0)).sum()),
                          pct_q05_same_dir_as_step1=100 * (same.q < 0.05).sum() / max(len(g), 1), pct_abs_rho_ge_0_3=100 * (g.rho.abs() >= 0.3).sum() / max(len(g), 1),
                          share_signal_top10pct=share, top10_by_abs_rho=", ".join(f"{r.gene} ({r.rho:+.2f}{'*' if r.q < 0.05 else ''})" for r in top.itertuples())))
G2 = pd.concat(G2, ignore_index=True); G2.to_csv(os.path.join(OUT, "step2_gene_level_all_genes_8_pathways.csv"), index=False)
# per-pathway long table (gene listed under each pathway it belongs to) for the Supplementary Table
longrows = [dict(cohort=r.cohort, pathway=sh(s), gene=r.gene, rho=r.rho, p=r.p, q=r.q) for r in G2.itertuples() for s in APPROVED if r.gene in set(H[s])]
pd.DataFrame(longrows).to_csv(os.path.join(OUT, "step2_gene_level_by_pathway_long.csv"), index=False)
SUMM2 = pd.DataFrame(SUMM2); SUMM2.to_csv(os.path.join(OUT, "step2_pathway_summary.csv"), index=False)
for r in SUMM2.itertuples():
    say(f"  [{r.cohort}] {sh(r.pathway):28s} ({r.step1_direction[:3]}) genes {r.n_tested:3d}  mean {r.mean_rho:+.2f} median {r.median_rho:+.2f}  +/− {r.n_pos}/{r.n_neg}  q<0.05 +{r.n_q05_pos}/−{r.n_q05_neg}  |rho|>=0.3 {r.pct_abs_rho_ge_0_3:3.0f}%  top10% share {r.share_signal_top10pct:.2f}")
# proliferation marker set (reused, curated 7 genes)
T = pd.read_csv(os.path.join(PAR, "tables", "step3_targeted_marker_sets_summary.csv"))
prol = T[T.marker_set == "Proliferation"]; say("  Proliferation 7-gene marker set (curated, reused): " + "; ".join(f"{r.cohort} mean rho {r.mean_rho:+.2f} [{r.genes}]" for r in prol.itertuples()))
# figure: gene rho distributions, 8 pathways x 2 cohorts
fig, ax = plt.subplots(figsize=(13, 6.5))
for i, s in enumerate(APPROVED):
    for k, (coh, c) in enumerate([("TCGA", "#4575B4"), ("CPTAC-3", "#F46D43")]):
        v = G2[(G2.cohort == coh) & (G2.gene.isin(set(H[s])))].rho.values; y = i + (0.18 if k else -0.18)
        ax.boxplot(v, positions=[y], widths=0.3, vert=False, patch_artist=True, boxprops=dict(facecolor=c, alpha=.6), medianprops=dict(color="k"), flierprops=dict(markersize=2))
        ax.text(0.72, y, f"n={len(v)}", va="center", fontsize=7, color=c)
ax.set_yticks(range(len(APPROVED))); ax.set_yticklabels([f"{sh(s)} ({'+' if d == 'positive' else '−'} in Step 1)" for s, d in zip(SEL.pathway, SEL.step1_direction)], fontsize=8.5)
ax.axvline(0, color="k", lw=.6); ax.set_xlim(-0.75, 0.8); ax.set_xlabel("per-gene Spearman rho vs PC6  (blue TCGA n=495, orange CPTAC-3 n=179)")
ax.set_title("DRAFT Step 2 — gene-level correlations within the 8 Step-1 pathways (all member genes; BH over all unique genes per cohort)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "DRAFT_step2_gene_level_8_pathways.png"), dpi=160, bbox_inches="tight"); plt.close()

# ================= Step 3: ssGSEA (submitted implementation) for the same 8 sets =================
say("\nStep 3 — ssGSEA (gseapy.ssgsea, sample_norm_method='rank', min_size=2, NES) for the 8 approved sets; BH over 8 per cohort")
gs_upper = {s: [g.upper() for g in H[s]] for s in APPROVED}
R3 = []
for coh, (X, pc6) in DATA.items():
    expr = X.copy(); expr.index = expr.index.str.upper(); expr = expr[~expr.index.duplicated()]
    ss = gp.ssgsea(data=expr, gene_sets=gs_upper, outdir=None, sample_norm_method="rank", no_plot=True, min_size=2, verbose=False)
    sc = ss.res2d.pivot(index="Term", columns="Name", values="NES").T.astype(float)
    sc = sc.loc[pc6.index]; sc.to_csv(os.path.join(OUT, f"step3_ssgsea_NES_8_pathways_{coh.replace('-3','')}.csv"))
    rows = []
    for s in APPROVED:
        rho, p = spearmanr(pc6.values, sc[s].values); rows.append(dict(cohort=coh, pathway=s, n=len(pc6), rho=rho, p=p))
    df = pd.DataFrame(rows); df["q"] = multipletests(df["p"], method="fdr_bh")[1]; df["sig"] = df.q.map(lambda q: "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns")
    R3.append(df)
R3 = pd.concat(R3, ignore_index=True); R3["pathway_short"] = R3.pathway.map(sh); R3.to_csv(os.path.join(OUT, "step3_ssgsea_vs_PC6_8_pathways.csv"), index=False)
for s in APPROVED:
    a = R3[(R3.cohort == "TCGA") & (R3.pathway == s)].iloc[0]; b = R3[(R3.cohort == "CPTAC-3") & (R3.pathway == s)].iloc[0]
    say(f"  {sh(s):28s} TCGA r={a.rho:+.3f} q={a.q:.1e} {a.sig:3s} | CPTAC-3 r={b.rho:+.3f} q={b.q:.1e} {b.sig}")
# comparison with the 5-marker-set ssGSEA (submitted implementation, FINAL v2) — kept for reference only
M5 = pd.read_csv(os.path.join(R1, "final_v2", "outputs", "results_ssgsea_pc6.csv")); M5.to_csv(os.path.join(OUT, "reference_step3_five_marker_sets_ssgsea_FINALv2.csv"), index=False)
# figure: ssGSEA bars
fig, ax = plt.subplots(figsize=(10, 5.5)); y = np.arange(len(APPROVED))
a = R3[R3.cohort == "TCGA"].set_index("pathway").loc[APPROVED]; b = R3[R3.cohort == "CPTAC-3"].set_index("pathway").loc[APPROVED]
ax.barh(y + 0.2, a.rho, height=0.4, color="#4575B4", label="TCGA (n = 495)"); ax.barh(y - 0.2, b.rho, height=0.4, color="#F46D43", label="CPTAC-3 (n = 179)")
for i, s in enumerate(APPROVED):
    for yy, r in [(i + 0.2, a.loc[s]), (i - 0.2, b.loc[s])]:
        ax.text(r.rho + (0.01 if r.rho >= 0 else -0.01), yy, f"{r.rho:+.2f} {r.sig}", va="center", ha="left" if r.rho >= 0 else "right", fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels([sh(s) for s in APPROVED], fontsize=9); ax.invert_yaxis(); ax.axvline(0, color="k", lw=.6); ax.set_xlim(-0.6, 0.75)
ax.set_xlabel("Spearman rho (ssGSEA NES vs PC6 z-score)"); ax.legend(loc="lower right", fontsize=8)
ax.set_title("DRAFT Step 3 — ssGSEA (gseapy, rank normalisation) for the 8 Step-1 pathways; BH over 8 sets per cohort", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "DRAFT_step3_ssgsea_8_pathways.png"), dpi=160, bbox_inches="tight"); plt.close()

# ================= Supplementary (kept separate): within-type values for the 8 sets, from the existing Step-5 tables =================
W5 = pd.read_csv(os.path.join(PAR, "tables", "step5_within_type_summary.csv")); W5 = W5[W5.pathway.isin(APPROVED)]
W5.to_csv(os.path.join(OUT, "supplementary_within_type_8_pathways_from_step5.csv"), index=False)
say("\nSupplementary (not part of the three Methods analyses): within-cancer-type values for the 8 sets (existing Step-5 tables, no rerun)")
for s in APPROVED:
    a = W5[(W5.cohort == "TCGA") & (W5.pathway == s)].iloc[0]; b = W5[(W5.cohort == "CPTAC-3") & (W5.pathway == s)].iloc[0]
    say(f"  {sh(s):28s} TCGA pooled {a.pooled_rho:+.2f} -> within {a.weighted_within_rho:+.2f} ({a.n_types_same_sign_as_pooled}/{a.n_types} types same sign) | CPTAC pooled {b.pooled_rho:+.2f} -> within {b.weighted_within_rho:+.2f} ({b.n_types_same_sign_as_pooled}/{b.n_types})")
open(os.path.join(HERE, "run.log"), "w", encoding="utf-8").write("\n".join(log) + "\n"); print("\nDONE")
