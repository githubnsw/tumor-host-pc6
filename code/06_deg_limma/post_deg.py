"""A (cont.): cancer-type-adjusted versions of Fig 4C (signature heatmap) and Fig 4D (marker violins) for the
within-cancer-type PC6 x SMA groups, plus a volcano of the adjusted limma result and a DEG summary.
Adjustment = within-cancer-type z-score of each gene (removes cancer-type mean/scale) before scoring; tests = Kruskal–Wallis
on the adjusted values and an F-test from the linear model expression ~ group + cancer_type (unadjusted values)."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import kruskal
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

HERE = os.path.dirname(os.path.abspath(__file__))
expr = pd.read_csv(os.path.join(HERE, "expr_log2_tcga.csv"), index_col=0); g = pd.read_csv(os.path.join(HERE, "groups_tcga.csv")).set_index("pid").loc[expr.columns]
G4 = ["PC6-Low / SMA-High", "PC6-Low / SMA-Low", "PC6-High / SMA-High", "PC6-High / SMA-Low"]
COL = {"PC6-Low / SMA-High": "#2166AC", "PC6-Low / SMA-Low": "#74ADD1", "PC6-High / SMA-High": "#F46D43", "PC6-High / SMA-Low": "#D73027"}
SIGS = {"Glycolysis": "HK2,PKM,ALDOA,LDHA,GAPDH,ENO1", "Hypoxia": "CA9,NDRG1,VEGFA,HIF1A,ADM", "Lactate/MCT": "LDHB,SLC16A3,SLC16A1", "CD8 T cell": "CD8A,CD8B,GZMA,GZMB,PRF1",
        "T cell Exhaustion": "HAVCR2,PDCD1,LAG3,TOX", "Treg": "FOXP3,IL2RA,IKZF2", "M2 Macrophage": "MSR1,CD163,MRC1", "IFN-γ Response": "STAT1,IRF1,CXCL9,CXCL10,IDO1",
        "Cachexia Signal": "MSTN,FBXO32,IL6,ACVR2B", "TGF-β": "TGFB1,TGFB2,SMAD2,SMAD3"}
SIGS = {k: [x for x in v.split(",") if x in expr.index] for k, v in SIGS.items()}
MARKERS = [("LDHA", "Glycolysis"), ("SLC16A3", "MCT4"), ("HAVCR2", "T Exhaustion"), ("FOXP3", "Treg"), ("MSTN", "Cachexia")]
grp = g["group4_withintype"]; ct = g["cancer_type"]
star = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def within_type_z(v):  # v: Series over patients
    return v.groupby(ct).transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))


def lm_group_p(v):  # F-test for group after cancer type
    d = pd.DataFrame({"y": v.values, "group": grp.values, "ct": ct.values})
    m = smf.ols("y ~ C(group) + C(ct)", data=d).fit(); return float(anova_lm(m, typ=2).loc["C(group)", "PR(>F)"])


# ── 4C: signature scores ──
rows, pk, pf, raw_rows = {}, {}, {}, {}
for name, genes in SIGS.items():
    score = expr.loc[genes].mean(axis=0)                     # per-patient mean log2 over signature genes (as in the cell: mean over genes & patients)
    adj = within_type_z(score)
    means = adj.groupby(grp).mean().reindex(G4); rows[name] = (means - means.mean()) / (means.std(ddof=0) + 1e-9)
    raw_m = score.groupby(grp).mean().reindex(G4); raw_rows[name] = (raw_m - raw_m.mean()) / (raw_m.std(ddof=0) + 1e-9)
    pk[name] = kruskal(*[adj[grp == q].values for q in G4]).pvalue; pf[name] = lm_group_p(score)
H = pd.DataFrame(rows).T; Hraw = pd.DataFrame(raw_rows).T
pd.DataFrame({"kruskal_p_adjusted_values": pk, "lm_F_p_group_after_cancer_type": pf}).to_csv(os.path.join(HERE, "fig4c_adjusted_tests.csv"), encoding="utf-8-sig")
H.to_csv(os.path.join(HERE, "fig4c_adjusted_heatmap_z.csv"), encoding="utf-8-sig")

# ── 4D: markers ──
mk = []
for gene, cat in MARKERS:
    v = expr.loc[gene]; adj = within_type_z(v)
    mk.append(dict(gene=gene, category=cat, kruskal_p_adjusted=kruskal(*[adj[grp == q].values for q in G4]).pvalue, lm_F_p_after_cancer_type=lm_group_p(v),
                   **{f"mean_adj_z[{q}]": adj[grp == q].mean() for q in G4}))
MK = pd.DataFrame(mk); MK.to_csv(os.path.join(HERE, "fig4d_adjusted_tests.csv"), index=False, encoding="utf-8-sig")

# ── figure: C (adjusted heatmap, with unadjusted beside it) + D (adjusted violins) ──
fig = plt.figure(figsize=(15, 6.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.55)
for j, (M, ttl, ptab) in enumerate([(Hraw, "C  Signature heatmap — unadjusted\n(within-type groups; z across 4 group means)", None), (H, "C'  Signature heatmap — cancer-type adjusted\n(within-type z-scored expression)", pk)]):
    ax = fig.add_subplot(gs[0, j]); im = ax.imshow(M.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([q.replace(" / ", "\n") for q in G4], fontsize=8); ax.set_yticks(range(len(M)))
    ax.set_yticklabels([f"{n}  {star(ptab[n]) if ptab else ''}" for n in M.index], fontsize=8.5); ax.set_title(ttl, fontsize=9.5)
    for (yi, xi), val in np.ndenumerate(M.values): ax.text(xi, yi, f"{val:+.1f}", ha="center", va="center", fontsize=7, color="white" if abs(val) > 1.2 else "black")
cb = fig.colorbar(im, ax=fig.axes, fraction=0.015, pad=0.01); cb.set_label("z-score", fontsize=8)
ax = fig.add_subplot(gs[0, 2]); base = np.arange(len(MARKERS)) * 5.5
for mi, (gene, cat) in enumerate(MARKERS):
    adj = within_type_z(expr.loc[gene])
    for gi, q in enumerate(G4):
        vals = adj[grp == q].values; pos = base[mi] + gi
        vp = ax.violinplot([vals], positions=[pos], widths=0.85, showmedians=True, showextrema=False)
        for b in vp["bodies"]: b.set_facecolor(COL[q]); b.set_alpha(0.8 if q == "PC6-High / SMA-Low" else 0.45); b.set_edgecolor("none")
        vp["cmedians"].set_color(COL[q])
    p = MK.loc[MK.gene == gene, "kruskal_p_adjusted"].iloc[0]
    ax.text(base[mi] + 1.5, 3.2, star(p), ha="center", fontsize=10, color="#c0392b" if p < 0.05 else "#888", fontweight="bold")
    ax.text(base[mi] + 1.5, -3.6, f"{gene}\n({cat})", ha="center", fontsize=8.5)
ax.set_xticks([]); ax.set_ylabel("Expression, within-cancer-type z-score"); ax.set_ylim(-3.9, 3.7); ax.axhline(0, color="#bbb", lw=.6)
ax.set_title("D'  Marker expression — cancer-type adjusted\n(Kruskal–Wallis on within-type z-scores)", fontsize=9.5)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[q]) for q in G4], labels=G4, fontsize=7.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.22))
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.suptitle("Fig 4C–D with PC6 × SMA groups defined by within-cancer-type medians, before and after cancer-type adjustment (TCGA, n = 495)", fontsize=11)
fig.savefig(os.path.join(HERE, "fig4cd_adjusted.png"), dpi=170, bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig4cd_adjusted.tiff"), dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)

# ── volcano of the adjusted limma model + DEG overlap with signature genes ──
L = pd.read_csv(os.path.join(HERE, "limma_withintype_adj_cancer_type.csv"))
fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(L.logFC, -np.log10(L["adj.P.Val"]), s=4, color="#bbb")
s_ = L[L.sig]; ax.scatter(s_.logFC, -np.log10(s_["adj.P.Val"]), s=6, c=np.where(s_.logFC > 0, "#D73027", "#2166AC"))
for gene in ["LDHA", "SLC16A3", "ALDOA", "PKM", "CA9", "CD8A", "GZMB", "CXCL9", "FOXP3", "HAVCR2", "CD163", "TGFB1"]:
    if gene in set(L.gene):
        r = L[L.gene == gene].iloc[0]; ax.annotate(gene, (r.logFC, -np.log10(r["adj.P.Val"])), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
ax.axvline(-0.5, ls="--", color="#999", lw=.8); ax.axvline(0.5, ls="--", color="#999", lw=.8); ax.axhline(-np.log10(0.05), ls="--", color="#999", lw=.8)
ax.set_xlabel("log2FC (PC6-High/SMA-Low vs others), limma ~ group + cancer type"); ax.set_ylabel("−log10(FDR)")
ax.set_title(f"Cancer-type-adjusted DEG: {int(s_.logFC.gt(0).sum())} up / {int(s_.logFC.lt(0).sum())} down", fontsize=10)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig4b_adjusted_volcano.png"), dpi=170); plt.close(fig)
sig_genes = {k: v for k, v in SIGS.items()}
ov = []
for k, genes in sig_genes.items():
    sub = L[L.gene.isin(genes)]
    ov.append(dict(signature=k, genes=",".join(genes), DEG_up=",".join(sub[sub.sig & (sub.logFC > 0)].gene), DEG_down=",".join(sub[sub.sig & (sub.logFC < 0)].gene),
                   mean_logFC=sub.logFC.mean()))
pd.DataFrame(ov).to_csv(os.path.join(HERE, "deg_adjusted_signature_overlap.csv"), index=False, encoding="utf-8-sig")
S = pd.read_csv(os.path.join(HERE, "limma_summary.csv")); PT = pd.read_csv(os.path.join(HERE, "limma_per_type_summary.csv"))
M = ["# A. DEG re-examination (TCGA; limma on log2 RSEM, FDR<0.05 & |logFC|>=0.5)", "",
     "| analysis | DEG | up | down |", "|---|---|---|---|"] + [f"| {r.analysis} | {r.n_sig} | {r.n_up} | {r.n_down} |" for r in S.itertuples()] + \
    ["", "Submitted (Wilcoxon, cohort-wide groups): 824 (354 up / 470 down). Fig 4 cell re-run with within-type groups (Wilcoxon): 975 (38 up / 937 down).", "",
     "## Per cancer type (within-type groups; types with >= 8 patients per side)", "", "| type | worst / other | DEG | up | down |", "|---|---|---|---|---|"] + \
    [f"| {r.cancer_type} | {r.n_worst} / {r.n_other} | {r.n_sig} | {r.n_up} | {r.n_down} |" for r in PT.itertuples()] + \
    ["", "## Signature genes in the adjusted DEG list", "", "| signature | up | down | mean logFC |", "|---|---|---|---|"] + \
    [f"| {r.signature} | {r.DEG_up or '–'} | {r.DEG_down or '–'} | {r.mean_logFC:+.2f} |" for r in pd.DataFrame(ov).itertuples()] + \
    ["", "## Fig 4C signatures (within-type groups): Kruskal on adjusted values / F-test group after cancer type", "", "| signature | KW p (adjusted) | F p (group | cancer type) | adjusted group z: " + " / ".join(G4) + " |", "|---|---|---|---|"] + \
    [f"| {n} | {pk[n]:.2g} | {pf[n]:.2g} | " + " / ".join(f"{H.loc[n, q]:+.1f}" for q in G4) + " |" for n in H.index] + \
    ["", "## Fig 4D markers", "", "| gene | KW p (adjusted) | F p | adjusted mean z: " + " / ".join(G4) + " |", "|---|---|---|---|"] + \
    [f"| {r.gene} | {r.kruskal_p_adjusted:.2g} | {r.lm_F_p_after_cancer_type:.2g} | " + " / ".join(f"{getattr(r, f'_{i+5}') if False else r[4 + i]:+.2f}" for i in range(4)) + " |" for r in MK.itertuples(index=False)]
open(os.path.join(HERE, "SUMMARY_A.md"), "w", encoding="utf-8").write("\n".join(M) + "\n"); print("\n".join(M))
