"""Biological Characterization of PC6 — Revised Analysis Plan (co-author, 2026-11-03). Steps 1–5.
Frozen inputs (FINAL v2): gene weights from the submitted weight-based cell (final_v2/outputs/gene_weights_all_PCs.csv =
pca10_components @ W3 @ W2 @ W1, identical to the 2026-04-03 submitted file), PC6 scores (ae_latent_pca10_all / _external),
TCGA RNA_Group1-4 (PanCanAtlas RSEM) and CPTAC ex_RNA_Group1-4 v2 (STAR TPM), official MSigDB h.all.v2023.2.Hs.symbols.gmt.
Nothing in the submitted figure folders is overwritten; all outputs go to this folder.
Step 1  weight-based Hallmark-50 enrichment (original method: universe = 3,000 AE genes, >=5 overlap, two-sided ranksums, BH per PC)
Step 2  ssGSEA (own implementation, Barbie 2009 alpha 0.25, validated vs GSVA rho 0.996) for 50 Hallmark sets, Spearman vs PC6, BH per cohort
Step 3  gene-level Spearman within pathways selected by a transparent rule + the original five marker sets (targeted, separately labelled)
Step 4  original five-marker-set ssGSEA (gseapy result file from the FINAL v2 run, copied as-is)
Step 5  within-cancer-type Spearman (types with n >= 20) for all 50 Hallmark sets, n-weighted summary
"""
import os, shutil, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ranksums, spearmanr
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__)); R1 = os.path.dirname(HERE); D = os.path.dirname(R1)
FIN = os.path.join(R1, "final_v2", "inputs"); OUT = os.path.join(HERE, "tables"); FIG = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
GMT = os.path.join(R1, "r2_immune_programs_sma", "sources", "h.all.v2023.2.Hs.symbols.gmt")
PCS = [f"PC{i}" for i in range(1, 11)]
MIN_OVERLAP = 5  # original cell-9 threshold
MARKER = {"Glycolysis": ["HK2", "PFKP", "ALDOA", "ENO1", "PGK1", "PKM", "LDHA", "GAPDH", "TPI1", "PGAM1"],
          "Hypoxia": ["VEGFA", "NDRG1", "CA9", "HIF1A", "ADM", "HMOX1"], "Lactate/MCT": ["LDHA", "LDHB", "SLC16A3", "SLC16A1"],
          "OXPHOS": ["SDHA", "SDHB", "COX5A", "NDUFB3", "UQCRC1"], "Proliferation": ["E2F1", "CDK4", "CCND1", "CDK1", "CCNA2", "TOP2A", "MKI67"]}
log = []
def say(s): print(s); log.append(s)

# ---------------- gene sets ----------------
H = {}
for line in open(GMT, encoding="utf-8"):
    p = line.rstrip("\n").split("\t"); H[p[0]] = [g for g in p[2:] if g]
short = lambda s: s.replace("HALLMARK_", "")
say(f"Official MSigDB Hallmark: {len(H)} sets, sizes {min(map(len, H.values()))}-{max(map(len, H.values()))} (file: {os.path.basename(GMT)})")

# ---------------- expression + PC6 ----------------
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
        cl = pd.read_csv(os.path.join(FIN, "TCGA_with_genes.csv")); cl["pid"] = cl["patient_id"].astype(str).str[:12]
    else:
        X = load_expr([os.path.join(FIN, f) for f in ["ex_RNA_Group1_LowMuscle_LowImmune.csv", "ex_RNA_Group2_LowMuscle_HighImmune.csv",
                                                     "ex_RNA_Group3_HighMuscle_LowImmune.csv", "ex_RNA_Group4_HighMuscle_HighImmune.csv"]], False)
        pc = pd.read_csv(os.path.join(FIN, "ae_latent_pca10_external.csv")); pc["pid"] = pc["patient_id"].astype(str)
        cl = pd.read_csv(os.path.join(FIN, "External_with_genes.csv")); cl["pid"] = cl["patient_id"].astype(str)
    pc = pc.drop_duplicates("pid"); pc["PC6_z"] = (pc["AE_PC6"] - pc["AE_PC6"].mean()) / pc["AE_PC6"].std()  # as in the Fig 3 cell
    common = [p for p in pc["pid"] if p in X.columns]
    X = X[common]; pc6 = pc.set_index("pid").loc[common, "PC6_z"].astype(float)
    ct = cl.drop_duplicates("pid").set_index("pid")["cancer_type"].reindex(common)
    say(f"[{coh}] expression {X.shape[0]} genes x {X.shape[1]} patients; PC6 matched n = {len(common)}; cancer types: {ct.value_counts().to_dict()}")
    return X, pc6, ct
DATA = {c: load_cohort(c) for c in ["TCGA", "CPTAC-3"]}

# ================= Step 1: weight-based enrichment, official Hallmark 50 =================
W = pd.read_csv(os.path.join(R1, "final_v2", "outputs", "gene_weights_all_PCs.csv"), index_col=0); universe = list(W.index)
say(f"\nStep 1: universe = {len(universe)} AE input genes; threshold >= {MIN_OVERLAP} overlapping genes (original)")
rows, untestable = [], []
for name, genes in H.items():
    present = [g for g in genes if g in W.index]
    if len(present) < MIN_OVERLAP:
        untestable.append((name, len(present))); continue
    pres = set(present); out_idx = [g for g in universe if g not in pres]
    for pc in PCS:
        in_w = W.loc[present, pc].values; out_w = W.loc[out_idx, pc].values
        stat, p = ranksums(in_w, out_w)
        rows.append(dict(PC=pc, pathway=name, n_overlap=len(present), n_set=len(genes), mean_weight=in_w.mean(), median_weight=np.median(in_w),
                         mean_weight_nonmember=out_w.mean(), direction="positive" if in_w.mean() > 0 else "negative", statistic=stat, p=p))
E1 = pd.DataFrame(rows)
E1 = pd.concat([g.assign(q=multipletests(g["p"], method="fdr_bh")[1]) for _, g in E1.groupby("PC", sort=False)], ignore_index=True)
E1.to_csv(os.path.join(OUT, "step1_weight_enrichment_all_PCs_hallmark50.csv"), index=False)
pd.DataFrame(untestable, columns=["pathway", "n_overlap"]).assign(status=f"not tested (< {MIN_OVERLAP} genes in AE input); NOT a nonsignificant result").to_csv(os.path.join(OUT, "step1_untestable_pathways.csv"), index=False)
n_test = E1[E1.PC == "PC6"].shape[0]
say(f"  testable sets: {n_test}/50; untestable: {len(untestable)} -> {[f'{short(n)}({k})' for n, k in untestable]}")
p6 = E1[E1.PC == "PC6"].sort_values("p"); p6.to_csv(os.path.join(OUT, "step1_PC6_summary.csv"), index=False)
say(f"  PC6: {int((p6.q < 0.05).sum())} sets q<0.05 of {n_test}")
for _, r in p6[p6.q < 0.05].sort_values("mean_weight", ascending=False).iterrows():
    say(f"    {short(r.pathway):36s} n={r.n_overlap:3d}/{r.n_set}  mean w {r.mean_weight:+.3f}  median {r.median_weight:+.3f}  q={r.q:.2e}  {r.direction}")
# comparison with the submitted 19-list run (same method)
old = pd.read_csv(os.path.join(R1, "final_v2", "outputs", "pathway_enrichment_all_PCs.csv")); o6 = old[old.PC == "PC6"]
o6.assign(version="submitted_19_typed_lists").to_csv(os.path.join(OUT, "step1_PC6_submitted_19lists_for_comparison.csv"), index=False)
# heatmap draft (testable sets only; signed -log10 q, capped 10)
sets_t = sorted(E1.pathway.unique(), key=lambda s: -E1[(E1.PC == "PC6") & (E1.pathway == s)].mean_weight.iloc[0])
M = np.zeros((len(sets_t), 10))
for i, s in enumerate(sets_t):
    for j, pc in enumerate(PCS):
        r = E1[(E1.PC == pc) & (E1.pathway == s)].iloc[0]; v = -np.log10(max(r.q, 1e-10)); M[i, j] = v if r.mean_weight > 0 else -v
fig, ax = plt.subplots(figsize=(11, 14)); vmax = np.abs(M).max()
im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), aspect="auto")
ax.set_xticks(range(10)); ax.set_xticklabels(PCS, fontweight="bold"); ax.set_yticks(range(len(sets_t))); ax.set_yticklabels([short(s) for s in sets_t], fontsize=8)
for i, s in enumerate(sets_t):
    for j, pc in enumerate(PCS):
        q = E1[(E1.PC == pc) & (E1.pathway == s)].q.iloc[0]
        if q < 0.001: ax.text(j, i, "**", ha="center", va="center", fontsize=9, fontweight="bold")
        elif q < 0.05: ax.text(j, i, "*", ha="center", va="center", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.5, label="signed -log10(q)  (sign = mean member weight)")
ax.set_title(f"DRAFT Step 1 — weight-based enrichment (linear approximation PCA·W3·W2·W1), official MSigDB Hallmark\n{n_test} testable sets (>= {MIN_OVERLAP} of the 3,000 AE genes); {len(untestable)} sets untestable, not shown; rows sorted by PC6 mean weight; * q<0.05 ** q<0.001", fontsize=9.5)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "DRAFT_step1_weight_enrichment_heatmap.png"), dpi=160, bbox_inches="tight"); plt.close()

# ================= Step 2: ssGSEA Hallmark 50 vs PC6 =================
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
say("\nStep 2: ssGSEA (own implementation, alpha 0.25, min-max) for 50 Hallmark sets")
S2, SC = {}, {}
for coh, (X, pc6, ct) in DATA.items():
    S = ssgsea(X, H); S.to_csv(os.path.join(OUT, f"step2_ssgsea_hallmark50_scores_{coh.replace('-3','')}.csv")); SC[coh] = S
    cov = {k: int(pd.Index(X.index).isin(v).sum()) for k, v in H.items()}
    rr = []
    for s in S.columns:
        rho, p = spearmanr(pc6.values, S[s].values); rr.append(dict(cohort=coh, pathway=s, n=len(pc6), genes_in_matrix=cov[s], rho=rho, p=p))
    df = pd.DataFrame(rr); df["q"] = multipletests(df["p"], method="fdr_bh")[1]; S2[coh] = df
    say(f"  [{coh}] n={len(pc6)}; sets q<0.05: {int((df.q < 0.05).sum())}/50 (positive {int(((df.q < 0.05) & (df.rho > 0)).sum())}, negative {int(((df.q < 0.05) & (df.rho < 0)).sum())})")
A2 = pd.concat(S2.values(), ignore_index=True); A2.to_csv(os.path.join(OUT, "step2_ssgsea_hallmark50_vs_PC6.csv"), index=False)
C2 = S2["TCGA"].set_index("pathway")[["rho", "q"]].join(S2["CPTAC-3"].set_index("pathway")[["rho", "q"]], lsuffix="_TCGA", rsuffix="_CPTAC")
C2["consistent"] = (C2.q_TCGA < 0.05) & (C2.q_CPTAC < 0.05) & (np.sign(C2.rho_TCGA) == np.sign(C2.rho_CPTAC))
C2["mean_rho"] = (C2.rho_TCGA + C2.rho_CPTAC) / 2; C2["rank_TCGA"] = C2.rho_TCGA.rank(ascending=False).astype(int); C2["rank_CPTAC"] = C2.rho_CPTAC.rank(ascending=False).astype(int)
C2 = C2.sort_values("mean_rho", ascending=False); C2.to_csv(os.path.join(OUT, "step2_cross_cohort_comparison.csv"))
rr_cc = spearmanr(C2.rho_TCGA, C2.rho_CPTAC)[0]
say(f"  cross-cohort: Spearman of the 50 rho values = {rr_cc:.2f}; consistent (q<0.05 both, same sign): {int(C2.consistent.sum())} "
    f"(positive {int((C2.consistent & (C2.mean_rho > 0)).sum())}, negative {int((C2.consistent & (C2.mean_rho < 0)).sum())})")
for s, r in C2[C2.consistent].iterrows():
    say(f"    {short(s):36s} TCGA {r.rho_TCGA:+.2f} (#{r.rank_TCGA}) q={r.q_TCGA:.1e} | CPTAC {r.rho_CPTAC:+.2f} (#{r.rank_CPTAC}) q={r.q_CPTAC:.1e}")
# agreement with the earlier Enrichr-2020 version (r2_hallmark_broad pooled PC6), if present
try:
    B = pd.read_csv(os.path.join(R1, "r2_hallmark_broad", "hallmark_associations.csv")); B = B[(B.target == "PC6") & (B.scope == "pooled")]
    for coh in ["TCGA", "CPTAC-3"]:
        b = B[B.cohort == coh].set_index("hallmark")["rho"]; a = S2[coh].set_index("pathway")["rho"]; a.index = [short(s) for s in a.index]
        com = [k for k in a.index if k in b.index]; say(f"  agreement with Enrichr-2020 version ({coh}, {len(com)} sets): rho-of-rho {spearmanr(a[com], b[com])[0]:.3f}")
except Exception as e: say(f"  (Enrichr-2020 comparison skipped: {e})")
# figure: scatter + bars
fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=[1, 1.1]))
ax = axes[0]; ax.axhline(0, color="#bbb", lw=.7); ax.axvline(0, color="#bbb", lw=.7)
col = ["#D73027" if (c and m > 0) else "#2166AC" if (c and m < 0) else "#999" for c, m in zip(C2.consistent, C2.mean_rho)]
ax.scatter(C2.rho_TCGA, C2.rho_CPTAC, c=col, s=36, edgecolor="white", lw=.5)
for s, r in C2.iterrows():
    if r.consistent or abs(r.rho_TCGA) > 0.3 or abs(r.rho_CPTAC) > 0.3: ax.annotate(short(s), (r.rho_TCGA, r.rho_CPTAC), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
ax.set_xlabel("Spearman rho vs PC6, TCGA (n = 495)"); ax.set_ylabel("Spearman rho vs PC6, CPTAC-3 (n = 179)")
ax.set_title(f"DRAFT Step 2 — Hallmark ssGSEA vs PC6, both cohorts (rho-of-rho {rr_cc:.2f})\nred/blue = q<0.05 in both cohorts, same sign", fontsize=10)
ax = axes[1]; top = pd.concat([C2.head(12), C2.tail(8)]); y = np.arange(len(top))
ax.barh(y + 0.2, top.rho_TCGA, height=0.4, color="#4575B4", label="TCGA"); ax.barh(y - 0.2, top.rho_CPTAC, height=0.4, color="#F46D43", label="CPTAC-3")
ax.set_yticks(y); ax.set_yticklabels([short(s) + (" ‡" if c else "") for s, c in zip(top.index, top.consistent)], fontsize=8); ax.invert_yaxis(); ax.axvline(0, color="k", lw=.6)
ax.set_xlabel("Spearman rho vs PC6"); ax.legend(loc="lower right", fontsize=8); ax.set_title("top 12 positive / 8 negative by mean rho (‡ consistent)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "DRAFT_step2_hallmark_ssgsea_vs_PC6.png"), dpi=160, bbox_inches="tight"); plt.close()

# ================= Step 3: gene-level within selected pathways =================
say("\nStep 3: pathway-selection rule = q<0.05 in BOTH cohorts with the same sign in Step 2 (ssGSEA) AND PC6 q<0.05 with the same sign in Step 1 (weight)")
w6 = E1[E1.PC == "PC6"].set_index("pathway")
core = [s for s in C2[C2.consistent].index if s in w6.index and w6.loc[s, "q"] < 0.05 and np.sign(w6.loc[s, "mean_weight"]) == np.sign(C2.loc[s, "mean_rho"])]
sec = [s for s in C2[C2.consistent].index if s not in core]
say(f"  core (both layers): {len(core)} -> {[short(s) for s in core]}"); say(f"  ssGSEA-consistent only: {len(sec)} -> {[short(s) for s in sec]}")
sel = core + sec
pd.DataFrame(dict(pathway=sel, tier=["core (Step1+Step2)"] * len(core) + ["Step2-consistent only"] * len(sec))).to_csv(os.path.join(OUT, "step3_selected_pathways.csv"), index=False)
def gene_corr(X, pc6, genes):
    out = []
    for g in genes:
        if g not in X.index: continue
        x = X.loc[g].values.astype(float)
        if np.nanstd(x) == 0: continue
        rho, p = spearmanr(pc6.values, x); out.append(dict(gene=g, rho=rho, p=p))
    return pd.DataFrame(out)
G3, SUMM3 = [], []
for coh, (X, pc6, ct) in DATA.items():
    frames = []
    for s in sel:
        gdf = gene_corr(X, pc6, H[s]); gdf["pathway"] = s; gdf["cohort"] = coh; frames.append(gdf)
    gall = pd.concat(frames, ignore_index=True); gall["q"] = multipletests(gall["p"], method="fdr_bh")[1]  # family = all genes tested in this cohort
    G3.append(gall)
    for s in sel:
        g = gall[gall.pathway == s]; sgn = np.sign(C2.loc[s, "mean_rho"]); same = g[np.sign(g.rho) == sgn]
        top = g.reindex(g.rho.abs().sort_values(ascending=False).index).head(5)
        pos = g.rho * sgn; share_top10 = pos.sort_values(ascending=False).head(max(1, int(np.ceil(0.1 * len(g))))).clip(lower=0).sum() / max(pos.clip(lower=0).sum(), 1e-9)
        SUMM3.append(dict(cohort=coh, pathway=s, tier="core" if s in core else "ssGSEA-only", n_genes_tested=len(g), median_rho=g.rho.median(), IQR_lo=g.rho.quantile(.25), IQR_hi=g.rho.quantile(.75),
                          pct_q05_same_sign=100 * ((same.q < 0.05).sum()) / max(len(g), 1), pct_abs_rho_ge_0_3_same_sign=100 * (same.rho.abs() >= 0.3).sum() / max(len(g), 1),
                          share_of_signal_in_top10pct_genes=share_top10, top5=", ".join(f"{r.gene} ({r.rho:+.2f})" for r in top.itertuples()),
                          ssgsea_rho=S2[coh].set_index("pathway").loc[s, "rho"]))
G3 = pd.concat(G3, ignore_index=True); G3.to_csv(os.path.join(OUT, "step3_gene_level_selected_pathways.csv"), index=False)
SUMM3 = pd.DataFrame(SUMM3); SUMM3.to_csv(os.path.join(OUT, "step3_gene_level_summary.csv"), index=False)
for _, r in SUMM3.iterrows():
    say(f"  [{r.cohort}] {short(r.pathway):34s} genes {r.n_genes_tested:3d}  median rho {r.median_rho:+.2f} (IQR {r.IQR_lo:+.2f}..{r.IQR_hi:+.2f})  q<0.05 same sign {r.pct_q05_same_sign:4.0f}%  |rho|>=0.3 {r.pct_abs_rho_ge_0_3_same_sign:3.0f}%  top10% share {r.share_of_signal_in_top10pct_genes:.2f}  ssGSEA rho {r.ssgsea_rho:+.2f}")
# targeted: original five marker sets (same per-gene method as the Fig 3 cell; BH over the 31 marker genes per cohort)
T3 = []
for coh, (X, pc6, ct) in DATA.items():
    genes = sorted({g for v in MARKER.values() for g in v}); gdf = gene_corr(X, pc6, genes); gdf["q"] = multipletests(gdf["p"], method="fdr_bh")[1]; gdf["cohort"] = coh
    for s, gl in MARKER.items():
        sub = gdf[gdf.gene.isin(gl)]; T3.append(dict(cohort=coh, marker_set=s, n=len(sub), mean_rho=sub.rho.mean(), median_rho=sub.rho.median(), genes=", ".join(f"{r.gene} ({r.rho:+.2f}{'*' if r.q < 0.05 else ''})" for r in sub.itertuples())))
    gdf.to_csv(os.path.join(OUT, f"step3_targeted_marker_genes_{coh.replace('-3','')}.csv"), index=False)
T3 = pd.DataFrame(T3); T3.to_csv(os.path.join(OUT, "step3_targeted_marker_sets_summary.csv"), index=False)
say("  targeted five marker sets (mean rho; should equal Fig 3A): " + "; ".join(f"{r.cohort} {r.marker_set} {r.mean_rho:+.2f}" for r in T3.itertuples()))
# figure: gene-level distributions for core sets
if sel:
    show = sel[:12]; fig, ax = plt.subplots(figsize=(13, 0.55 * len(show) + 2))
    for i, s in enumerate(show):
        for k, (coh, c) in enumerate([("TCGA", "#4575B4"), ("CPTAC-3", "#F46D43")]):
            v = G3[(G3.cohort == coh) & (G3.pathway == s)].rho.values
            ax.boxplot(v, positions=[i + (0.18 if k else -0.18)], widths=0.3, vert=False, patch_artist=True, boxprops=dict(facecolor=c, alpha=.6), medianprops=dict(color="k"), flierprops=dict(markersize=2))
    ax.set_yticks(range(len(show))); ax.set_yticklabels([short(s) + (" (core)" if s in core else "") for s in show], fontsize=8); ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel("per-gene Spearman rho vs PC6 (blue TCGA, orange CPTAC-3)"); ax.set_title("DRAFT Step 3 — gene-level correlations within selected Hallmark sets", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "DRAFT_step3_gene_level_boxplots.png"), dpi=160, bbox_inches="tight"); plt.close()

# ================= Step 4: targeted ssGSEA (original gseapy run, FINAL v2) =================
src4 = os.path.join(R1, "final_v2", "outputs", "results_ssgsea_pc6.csv"); shutil.copy(src4, os.path.join(OUT, "step4_targeted_ssgsea_five_marker_sets_FINALv2.csv"))
R4 = pd.read_csv(src4); say("\nStep 4 (copied, gseapy original): " + "; ".join(f"{r.cohort} {r.pathway} {r.r:+.2f} ({r.sig})" for r in R4.itertuples()))

# ================= Step 5: within-cancer-type =================
say("\nStep 5: within-cancer-type Spearman (types n >= 20), all 50 Hallmark sets")
rows5, summ5 = [], []
for coh, (X, pc6, ct) in DATA.items():
    S = SC[coh]
    for s in S.columns:
        per = []
        for t, idx in ct.groupby(ct).groups.items():
            idx = [i for i in idx if i in S.index]
            if len(idx) < 20: continue
            rho, p = spearmanr(pc6.loc[idx].values, S.loc[idx, s].values); per.append((t, len(idx), rho, p)); rows5.append(dict(cohort=coh, pathway=s, cancer_type=t, n=len(idx), rho=rho, p=p))
        n = np.array([x[1] for x in per]); r = np.array([x[2] for x in per]); pooled = S2[coh].set_index("pathway").loc[s, "rho"]
        summ5.append(dict(cohort=coh, pathway=s, pooled_rho=pooled, n_types=len(per), weighted_within_rho=float((n * r).sum() / n.sum()), n_types_same_sign_as_pooled=int((np.sign(r) == np.sign(pooled)).sum()),
                          min_rho=r.min(), max_rho=r.max(), per_type="; ".join(f"{t} (n={k}) {v:+.2f}" for t, k, v, _ in per)))
P5 = pd.DataFrame(rows5); P5.to_csv(os.path.join(OUT, "step5_within_type_per_cancer_type.csv"), index=False)
M5 = pd.DataFrame(summ5); M5.to_csv(os.path.join(OUT, "step5_within_type_summary.csv"), index=False)
for s in sel:
    for coh in ["TCGA", "CPTAC-3"]:
        r = M5[(M5.cohort == coh) & (M5.pathway == s)].iloc[0]
        say(f"  [{coh}] {short(s):34s} pooled {r.pooled_rho:+.2f} -> within-type (weighted) {r.weighted_within_rho:+.2f}; same sign in {r.n_types_same_sign_as_pooled}/{r.n_types} types; range {r.min_rho:+.2f}..{r.max_rho:+.2f}")
# figure: within-type heatmap for selected sets
if sel:
    show = sel[:15]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.5 * len(show) + 2.5), gridspec_kw=dict(width_ratios=[7, 3]))
    for ax, coh in zip(axes, ["TCGA", "CPTAC-3"]):
        pv = P5[(P5.cohort == coh) & (P5.pathway.isin(show))].pivot(index="pathway", columns="cancer_type", values="rho").reindex(show)
        im = ax.imshow(pv.values, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
        ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels([f"{c}\n(n={int(P5[(P5.cohort==coh)&(P5.cancer_type==c)].n.iloc[0])})" for c in pv.columns], fontsize=8)
        ax.set_yticks(range(len(show))); ax.set_yticklabels([short(s) for s in show] if coh == "TCGA" else [""] * len(show), fontsize=8)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isfinite(v): ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.4 else "black")
        ax.set_title(f"{coh}: within-cancer-type Spearman rho vs PC6", fontsize=10)
    plt.colorbar(im, ax=axes, shrink=0.6, label="rho"); plt.savefig(os.path.join(FIG, "DRAFT_step5_within_type_heatmap.png"), dpi=160, bbox_inches="tight"); plt.close()

open(os.path.join(HERE, "run.log"), "w", encoding="utf-8").write("\n".join(log) + "\n")
json.dump(dict(core=core, ssgsea_only=sec, untestable=[u[0] for u in untestable], rho_of_rho=rr_cc), open(os.path.join(HERE, "selection.json"), "w"), indent=1)
print("\nDONE")
