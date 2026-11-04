"""C (Reviewer 2 #4) — unbiased gene-set view: Spearman correlation of PC1–PC10 with all 50 MSigDB Hallmark ssGSEA scores
(own ssGSEA implementation validated vs GSVA, rho 0.996; scores in seed_check/), in TCGA and CPTAC-3; plus GEP score vs the 50
hallmarks (justification of the immune axis). Pooled and within-cancer-type (n-weighted) versions for PC6 and GEP.
Outputs: hallmark50_pc_corr_{TCGA,CPTAC}.csv, hallmark50_heatmap.(png|tiff), pc6_gep_hallmark_bars.png, SUMMARY.md"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); R1 = r"C:\Users\nsw\Desktop\body_composition\revision_r1"; D = os.path.dirname(R1)
FIN = os.path.join(R1, "final_v2", "inputs"); SC = os.path.join(D, "seed_check")
PCS = [f"AE_PC{i}" for i in range(1, 11)]


def load(coh):
    if coh == "TCGA":
        hs = pd.read_csv(os.path.join(SC, "hallmark_ssgsea_TCGA_rsem.csv"), index_col=0); hs.index = hs.index.str[:12]
        pcs = pd.read_csv(os.path.join(FIN, "ae_latent_pca10_all.csv")); pcs["pid"] = pcs.patient_id.str[:12]
        clin = pd.read_csv(os.path.join(FIN, "TCGA_with_genes.csv")); clin["pid"] = clin.patient_id.str[:12]
    else:
        hs = pd.read_csv(os.path.join(R1, "final_v2", "hallmark_ssgsea_CPTAC_tpm_v2.csv"), index_col=0)
        pcs = pd.read_csv(os.path.join(FIN, "ae_latent_pca10_external.csv")); pcs["pid"] = pcs.patient_id.astype(str)
        clin = pd.read_csv(os.path.join(FIN, "External_with_genes.csv")); clin["pid"] = clin.patient_id.astype(str)
    hs = hs.loc[~hs.index.duplicated()]
    m = pcs.drop_duplicates("pid").set_index("pid")[PCS].join(clin.drop_duplicates("pid").set_index("pid")[["GEP_score", "cancer_type"]], how="inner")
    m = m.join(hs, how="inner"); return m, list(hs.columns)


res, summ = {}, ["# C. Hallmark 50 × latent components (R2-4)", ""]
for coh in ["TCGA", "CPTAC-3"]:
    m, H = load(coh)
    C = pd.DataFrame({pc: [spearmanr(m[pc], m[h])[0] for h in H] for pc in PCS + ["GEP_score"]}, index=H)
    # within-cancer-type (n-weighted) for PC6 and GEP
    for v in ["AE_PC6", "GEP_score"]:
        per = []
        for ct, g in m.groupby("cancer_type"):
            if len(g) < 20: continue
            per.append((len(g), np.array([spearmanr(g[v], g[h])[0] for h in H])))
        C[f"{v}_withintype"] = sum(n * r for n, r in per) / sum(n for n, r in per)
    C.to_csv(os.path.join(HERE, f"hallmark50_corr_{coh.replace('-3', '')}.csv"), encoding="utf-8-sig"); res[coh] = (C, len(m))
    top = C["AE_PC6"].sort_values(ascending=False)
    summ += [f"## {coh} (n = {len(m)})", "", "PC6 — top 8 positive (pooled ρ / within-type ρ):",
             "; ".join(f"{h} {top[h]:+.2f}/{C.loc[h, 'AE_PC6_withintype']:+.2f}" for h in top.index[:8]), "",
             "PC6 — top 5 negative: " + "; ".join(f"{h} {top[h]:+.2f}/{C.loc[h, 'AE_PC6_withintype']:+.2f}" for h in top.index[-5:]), "",
             "PC6 vs immune sets (pooled / within-type): " + "; ".join(f"{h} {C.loc[h, 'AE_PC6']:+.2f}/{C.loc[h, 'AE_PC6_withintype']:+.2f}" for h in ["INTERFERON_GAMMA_RESPONSE", "INTERFERON_ALPHA_RESPONSE", "INFLAMMATORY_RESPONSE", "ALLOGRAFT_REJECTION", "IL6_JAK_STAT3_SIGNALING", "TNFA_SIGNALING_VIA_NFKB"] if h in C.index), "",
             "PC6 rank of GLYCOLYSIS / HYPOXIA / OXIDATIVE_PHOSPHORYLATION among 50 (pooled): " + ", ".join(f"{h} #{int((top.index == h).argmax()) + 1} ({top[h]:+.2f})" for h in ["GLYCOLYSIS", "HYPOXIA", "OXIDATIVE_PHOSPHORYLATION"] if h in top.index), "",
             "GEP score — top 8 (pooled / within-type): " + "; ".join(f"{h} {C.loc[h, 'GEP_score']:+.2f}/{C.loc[h, 'GEP_score_withintype']:+.2f}" for h in C["GEP_score"].sort_values(ascending=False).index[:8]), ""]

# ── heatmap: hallmarks ordered by TCGA PC6 correlation; two cohorts side by side (PC1–10 + GEP) ──
Ct, Cc = res["TCGA"][0], res["CPTAC-3"][0]; order = Ct["AE_PC6"].sort_values(ascending=False).index
cols = PCS + ["GEP_score"]; labels = [c.replace("AE_", "") for c in PCS] + ["GEP"]
fig, axes = plt.subplots(1, 2, figsize=(13, 13), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, (coh, C) in zip(axes, [("TCGA", Ct), ("CPTAC-3", Cc)]):
    M = C.loc[order, cols]; im = ax.imshow(M.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, rotation=90, fontsize=9); ax.set_title(f"{coh} (n = {res[coh][1]})", fontsize=11)
    ax.axvline(len(PCS) - 0.5, color="k", lw=1); ax.axvline(4.5, color="w", lw=1.5); ax.axvline(5.5, color="w", lw=1.5)
    for (yi, xi), v in np.ndenumerate(M.values):
        if abs(v) >= 0.3: ax.text(xi, yi, f"{v:+.2f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.45 else "black")
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([h.replace("_", " ").title() for h in order], fontsize=8)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02); cb.set_label("Spearman ρ (ssGSEA score vs component)")
fig.suptitle("All 50 MSigDB Hallmark gene sets × latent components PC1–PC10 (and GEP score); rows ordered by TCGA PC6 correlation", fontsize=11, y=0.995)
fig.savefig(os.path.join(HERE, "hallmark50_heatmap.png"), dpi=160, bbox_inches="tight"); fig.savefig(os.path.join(HERE, "hallmark50_heatmap.tiff"), dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)

# ── bars: PC6 and GEP vs 50 hallmarks, pooled vs within-type, both cohorts ──
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
for i, v in enumerate(["AE_PC6", "GEP_score"]):
    for j, (coh, C) in enumerate([("TCGA", Ct), ("CPTAC-3", Cc)]):
        ax = axes[i, j]; o = C[v].sort_values(ascending=False).index; y = np.arange(len(o))[::-1]
        ax.barh(y + 0.2, C.loc[o, v], 0.4, color="#95a5a6", label="pooled"); ax.barh(y - 0.2, C.loc[o, f"{v}_withintype"], 0.4, color="#c0392b", label="within cancer type")
        ax.set_yticks(y); ax.set_yticklabels([h.replace("_", " ").title() for h in o], fontsize=6.5); ax.axvline(0, color="k", lw=.8)
        ax.set_title(f"{coh}: {'PC6' if v == 'AE_PC6' else 'GEP score'} vs 50 hallmarks (Spearman ρ)", fontsize=10); ax.set_xlim(-0.7, 0.9)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
axes[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "pc6_gep_hallmark_bars.png"), dpi=150); plt.close(fig)
open(os.path.join(HERE, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(summ) + "\n"); print("\n".join(summ))
