"""Manuscript Figure 4 and Supplementary Figure 9 (final layout).
Usage:  python code/05_figure4/make_fig4_final_layout.py   (lifelines, statsmodels, matplotlib)

  Main Figure 4:  A 4-group Kaplan-Meier (pooled + stratified log-rank p)   B cancer-type-stratified Cox forest (TCGA, CPTAC-3)
                  C transcriptional signature heatmap (within-cancer-type z)   D cancer-type-adjusted DEG volcano (limma ~ group + cancer type)
  Supplementary:  A marker violins (within-type z)   B Kaplan-Meier per cancer type   C signature differences per cancer type   D DEG counts, pooled vs adjusted
Group definition unchanged from the submission (cohort-wide medians). Same computations as run_fig4_all_stratified.py."""
import os, warnings, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import NullFormatter
from scipy.stats import kruskal, chi2, t as tdist
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
FIN = os.path.join(REPO, "data"); DEG = os.path.join(REPO, "data", "deg"); EXPR = os.path.join(REPO, "data", "expression", "tcga_expr_log2_495.csv.gz")
OUT = os.path.join(REPO, "results", "reproduced", "figure4"); os.makedirs(OUT, exist_ok=True)
G4 = ["PC6-Low / SMA-High", "PC6-Low / SMA-Low", "PC6-High / SMA-High", "PC6-High / SMA-Low"]; WORST = G4[3]
COL = {G4[0]: "#2166AC", G4[1]: "#74ADD1", G4[2]: "#F46D43", G4[3]: "#D73027"}
SHORT = {G4[0]: "PC6-L / SMA-H", G4[1]: "PC6-L / SMA-L", G4[2]: "PC6-H / SMA-H", G4[3]: "PC6-H / SMA-L"}
star = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
fp = lambda p: "p<0.001" if p < 0.001 else f"p={p:.3f}"
fe = lambda p: f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"
LS, TS, TIS, PS = 17, 14, 18, 22
NUM = []


def letter_at(ax, letter, x=-0.14, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PS, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=110, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".tiff"), dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)


def stage_bin(x):
    s = str(x).upper()
    if "[" in s or "UNKNOWN" in s or s.strip() in ("", "NAN", "NONE"): return np.nan
    if "III" in s or "IV" in s: return 1.0
    if "II" in s or "I" in s: return 0.0
    return np.nan


def load(coh):
    clin = pd.read_csv(os.path.join(FIN, "TCGA_with_genes.csv" if coh == "TCGA" else "External_with_genes.csv"))
    pcs = pd.read_csv(os.path.join(FIN, "ae_latent_pca10_all.csv" if coh == "TCGA" else "ae_latent_pca10_external.csv"))
    key = (lambda s: s.astype(str).str[:12]) if coh == "TCGA" else (lambda s: s.astype(str))
    clin["pid"], pcs["pid"] = key(clin.patient_id), key(pcs.patient_id)
    d = clin.merge(pcs[["pid", "AE_PC6"]].drop_duplicates("pid"), on="pid")
    d["PC6_z"] = (d.AE_PC6 - d.AE_PC6.mean()) / d.AE_PC6.std()
    d["OS_months"] = pd.to_numeric(d.OS_time, errors="coerce") / 30.44; d["OS"] = pd.to_numeric(d.OS, errors="coerce"); d["age"] = pd.to_numeric(d.age, errors="coerce")
    d["SMA_zscore"] = pd.to_numeric(d.SMA_zscore, errors="coerce"); d["sex_bin"] = (d.sex.astype(str).str.upper() == "MALE").astype(float); d["stage_bin"] = d.stage.apply(stage_bin)
    d = d.dropna(subset=["PC6_z", "SMA_zscore", "OS_months", "OS"]).copy()
    pm, sm = d.PC6_z.median(), d.SMA_zscore.median()
    d["group"] = np.where(d.PC6_z >= pm, "PC6-High", "PC6-Low") + " / " + np.where(d.SMA_zscore >= sm, "SMA-High", "SMA-Low")
    return d


def stratified_logrank(df, G="group", S="cancer_type"):
    groups = sorted(df[G].unique()); k = len(groups); OE = np.zeros(k); V = np.zeros((k, k))
    for _, d in df.groupby(S):
        for tt in np.sort(d.loc[d.OS == 1, "OS_months"].unique()):
            at = d[d.OS_months >= tt]; n = len(at); dead = d[(d.OS_months == tt) & (d.OS == 1)]; dt = len(dead)
            if n < 2 or dt == 0: continue
            nj = np.array([(at[G] == g).sum() for g in groups], float); dj = np.array([(dead[G] == g).sum() for g in groups], float)
            p = nj / n; OE += dj - dt * p; V += dt * (n - dt) / (n - 1) * (np.diag(p) - np.outer(p, p))
    oe, v = OE[:-1], V[:-1, :-1]; stat = float(oe @ np.linalg.pinv(v) @ oe); return stat, float(chi2.sf(stat, k - 1))


def cox_groups(d):
    u = d.dropna(subset=["age", "stage_bin"]); u = u[u.OS_months > 0].copy()
    adj = ["age", "sex_bin", "stage_bin"] if u.sex_bin.std() > 0.05 else ["age", "stage_bin"]
    for g in G4[1:]: u[g] = (u.group == g).astype(float)
    cph = CoxPHFitter().fit(u[["OS_months", "OS", "cancer_type"] + G4[1:] + adj], "OS_months", "OS", strata="cancer_type"); s = cph.summary
    out = pd.DataFrame([dict(group=g, n=int((u.group == g).sum()), HR=s.loc[g, "exp(coef)"], lo=s.loc[g, "exp(coef) lower 95%"], hi=s.loc[g, "exp(coef) upper 95%"], p=s.loc[g, "p"]) for g in G4[1:]])
    return out, len(u), int(u.OS.sum()), int((u.group == G4[0]).sum()), adj


data = {c: load(c) for c in ["TCGA", "CPTAC-3"]}
expr = pd.read_csv(EXPR, index_col=0, compression="gzip"); g = pd.read_csv(os.path.join(DEG, "groups_tcga.csv")).set_index("pid").loc[expr.columns]
grp, ct = g["group4_cohortwide"], g["cancer_type"]; assert grp.value_counts().to_dict() == {G4[2]: 131, G4[1]: 130, G4[0]: 117, G4[3]: 117}
wz = lambda v: v.groupby(ct).transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
SIGS = {"Glycolysis": "HK2,PKM,ALDOA,LDHA,GAPDH,ENO1", "Hypoxia": "CA9,NDRG1,VEGFA,HIF1A,ADM", "Lactate/MCT": "LDHB,SLC16A3,SLC16A1", "CD8 T cell": "CD8A,CD8B,GZMA,GZMB,PRF1",
        "T cell Exhaustion": "HAVCR2,PDCD1,LAG3,TOX", "Treg": "FOXP3,IL2RA,IKZF2", "M2 Macrophage": "MSR1,CD163,MRC1", "IFN-γ Response": "STAT1,IRF1,CXCL9,CXCL10,IDO1",
        "Cachexia Signal": "MSTN,FBXO32,IL6,ACVR2B", "TGF-β": "TGFB1,TGFB2,SMAD2,SMAD3"}
SIGS = {k: [x for x in v.split(",") if x in expr.index] for k, v in SIGS.items()}
TYPES = sorted(ct.unique(), key=lambda t: -int((ct == t).sum())); nworst = {t: int(((ct == t) & (grp == WORST)).sum()) for t in TYPES}; ntype = {t: int((ct == t).sum()) for t in TYPES}


def f_group_after_ct(v):
    dd = pd.DataFrame({"y": v.values, "group": grp.values, "ct": ct.values}); m = smf.ols("y ~ C(group) + C(ct)", data=dd).fit(); return float(anova_lm(m, typ=2).loc["C(group)", "PR(>F)"])


def worst_adj_effect(adjv):
    dd = pd.DataFrame({"y": adjv.values, "w": (grp == WORST).astype(float).values, "ct": ct.values}); m = smf.ols("y ~ w + C(ct)", data=dd).fit()
    return m.params["w"], m.conf_int().loc["w", 0], m.conf_int().loc["w", 1], m.pvalues["w"]


M, PK, PF, EFF, PT = {}, {}, {}, {}, {}
for name, genes in SIGS.items():
    score = expr.loc[genes].mean(axis=0); adj = wz(score)
    M[name] = adj.groupby(grp).mean().reindex(G4); PK[name] = kruskal(*[adj[grp == q].values for q in G4]).pvalue; PF[name] = f_group_after_ct(score); EFF[name] = worst_adj_effect(adj)
    PT[name] = {t: (adj[(ct == t) & (grp == WORST)].mean() - adj[(ct == t) & (grp != WORST)].mean()) if nworst[t] >= 3 else np.nan for t in TYPES}
M = pd.DataFrame(M).T[G4]; PTd = pd.DataFrame(PT).T[TYPES]
L = pd.read_csv(os.path.join(DEG, "limma_cohortwide_adj_cancer_type.csv")); Lu = pd.read_csv(os.path.join(DEG, "limma_cohortwide_unadjusted.csv"))


# ═════════════════════════════ panel drawers ═════════════════════════════
def panel_km(ax, letter="A"):
    d = data["TCGA"]; kmfs = []
    for gname in G4:
        sub = d[d.group == gname]; k = KaplanMeierFitter().fit(sub.OS_months, sub.OS, label=f"{SHORT[gname]} (n={len(sub)})")
        k.plot_survival_function(ax=ax, color=COL[gname], lw=2.6, ci_show=True, ci_alpha=0.10, show_censors=True, censor_styles={"ms": 5, "marker": "+"}); kmfs.append(k)
    lrp = multivariate_logrank_test(d.OS_months, d.group, d.OS).p_value; st_, lrs = stratified_logrank(d)
    xmax = int(np.ceil(d.OS_months.max() / 20.0)) * 20; ax.set_xlim(0, xmax); ax.set_xticks(np.arange(0, xmax + 1, 20))
    ax.set_xlabel("Time (months)", fontsize=LS); ax.set_ylabel("Overall Survival Probability", fontsize=LS); ax.tick_params(labelsize=TS)
    ax.set_title("PC6 × SMA 4-Group Survival (TCGA)\n(worst = PC6-High / SMA-Low)", fontsize=TIS, fontweight="bold", pad=12); letter_at(ax, letter)
    ax.spines[["top", "right"]].set_visible(False); ax.spines["left"].set_bounds(0, 1.05)
    ax.legend(fontsize=11, loc="upper right", bbox_to_anchor=(0.97, 0.97)).get_frame().set_edgecolor("#ccc")
    ax.text(xmax * 0.97, 0.06, f"Log-rank  p = {fe(lrp)}\nStratified by cancer type  p = {fe(lrs)}", fontsize=13, va="bottom", ha="right", fontweight="bold", color="#222",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF8E1", ec="#F39C12", lw=1.5, alpha=0.95))
    # number-at-risk table (data coordinates, as in the submitted figure)
    times = np.arange(0, xmax + 1, 20).astype(float); ax.set_ylim(-0.60, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axhspan(-0.55, -0.05, color="#F7F7F7", alpha=0.7, zorder=0); ax.axhline(0, color="#222", lw=1.3, alpha=0.9, zorder=2)
    ax.text(xmax / 2, -0.10, "Number at risk", ha="center", va="center", fontsize=13, fontweight="bold", color="#333"); ax.axhline(-0.16, xmin=0.005, xmax=0.995, color="#bbb", lw=0.8)
    row_y = [-0.24, -0.32, -0.40, -0.48]
    for gi, gname in enumerate(G4):
        sub = d[d.group == gname]; ax.text(-xmax * 0.04, row_y[gi], SHORT[gname], ha="right", va="center", fontsize=11, fontweight="bold", color=COL[gname])
        for t in times: ax.text(t, row_y[gi], str(int((sub.OS_months >= t).sum())), ha="center", va="center", fontsize=11, fontweight="bold", color=COL[gname])
    NUM.extend([dict(panel="4A", item="pooled log-rank p", value=lrp), dict(panel="4A", item="stratified log-rank p", value=lrs), dict(panel="4A", item="stratified chi2", value=st_)])


def panel_forest(ax, letter="B"):
    y = 0; ticks, labels, bold = [], [], []
    for coh in ["TCGA", "CPTAC-3"]:
        d = data[coh]; hrs, n, ev, nref, adj = cox_groups(d); st_, lp = stratified_logrank(d)
        ax.text(0.47, y, f"{coh}  (n = {n}, events = {ev}); stratified log-rank {fp(lp)}", fontsize=14, fontweight="bold", va="center", ha="left"); y -= 1.1
        ticks.append(y); labels.append(f"{SHORT[G4[0]]} (ref., n = {nref})"); bold.append(False); ax.plot(1, y, "s", color=COL[G4[0]], ms=10); y -= 1
        NUM.extend([dict(panel="4B", cohort=coh, item="stratified log-rank p", value=lp), dict(panel="4B", cohort=coh, item="n/events/ref", value=f"{n}/{ev}/{nref}")])
        for r in hrs.itertuples():
            ax.errorbar(r.HR, y, xerr=[[r.HR - r.lo], [r.hi - r.HR]], fmt="o", color=COL[r.group], ms=10, capsize=5, lw=2.4)
            ticks.append(y); labels.append(f"{SHORT[r.group]} (n = {r.n})"); bold.append(False)
            ax.text(12.5, y, f"{r.HR:.2f} ({r.lo:.2f}–{r.hi:.2f})  {fp(r.p)}", fontsize=14.5, va="center", ha="left", family="monospace", fontweight="bold" if r.p < 0.05 else "normal")
            NUM.append(dict(panel="4B", cohort=coh, item=f"{r.group} vs ref", value=f"{r.HR:.2f} ({r.lo:.2f}-{r.hi:.2f}) {fp(r.p)}")); y -= 1
        y -= 0.9
    ax.axvline(1, color="#999", lw=1.0, ls="--"); ax.set_xscale("log"); ax.set_xlim(0.45, 150); ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5", "1", "2", "4", "8"], fontsize=15); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=15)
    for lab, b in zip(ax.get_yticklabels(), bold): lab.set_fontweight("bold" if b else "normal")
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", which="minor", bottom=False); ax.set_ylim(y + 0.5, 0.9); ax.spines["bottom"].set_bounds(0.45, 10)
    ax.set_xlabel("Hazard ratio vs PC6-L / SMA-H (95% CI)\nCox stratified by cancer type; adjusted for age, sex, stage", fontsize=LS - 1)
    ax.set_title("Cancer-type-stratified Cox regression\n(same four subgroups)", fontsize=TIS, fontweight="bold", pad=12); letter_at(ax, letter)
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)


def panel_heatmap(ax, fig, letter="C"):
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([q.replace(" / ", "\n") for q in G4], fontsize=13)
    for lab, q in zip(ax.get_xticklabels(), G4): lab.set_color(COL[q]); lab.set_fontweight("bold")
    ax.set_yticks(range(len(M))); ax.set_yticklabels([f"{n}  {star(PK[n])}" for n in M.index], fontsize=13)
    for (yi, xi), val in np.ndenumerate(M.values): ax.text(xi, yi, f"{val:+.2f}", ha="center", va="center", fontsize=12, color="white" if abs(val) > 0.6 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label("group mean of within-cancer-type z-score", fontsize=12); cb.ax.tick_params(labelsize=11)
    ax.set_title("Transcriptional Signature Heatmap\n(within-cancer-type standardized)", fontsize=TIS, fontweight="bold", pad=12); letter_at(ax, letter, x=-0.30)
    ax.text(0.5, -0.16, "group means of within-cancer-type z-scores; stars: Kruskal–Wallis across the four subgroups", transform=ax.transAxes, ha="center", va="top", fontsize=11, color="#444", style="italic")
    for n in M.index: NUM.append(dict(panel="4C", item=f"{n} KW p / F p", value=f"{PK[n]:.2g} / {PF[n]:.2g}"))


def panel_volcano(ax, letter="D"):
    s_ = L[L.sig]; nu, nd = int((s_.logFC > 0).sum()), int((s_.logFC < 0).sum())
    ax.scatter(L.logFC, -np.log10(L["adj.P.Val"]), s=6, color="#c8c8c8", linewidths=0)
    ax.scatter(s_.logFC, -np.log10(s_["adj.P.Val"]), s=10, c=np.where(s_.logFC > 0, "#D73027", "#2166AC"), linewidths=0)
    Ls = L.set_index("gene"); OFF = {"CA9": (6, 4), "SLC16A3": (6, 4), "NDRG1": (6, -12), "PKM": (-34, 6), "ALDOA": (-44, -4), "LDHA": (6, 4), "FOXP3": (6, -12), "MRC1": (-48, 12), "CD163": (-48, -10), "MSTN": (10, -4), "HAVCR2": (6, -12), "CD8A": (-40, -4)}
    for gname, off in OFF.items():
        if gname in Ls.index:
            r = Ls.loc[gname]; c = ("#D73027" if r.logFC > 0 else "#2166AC") if r.sig else "#555"
            ax.annotate(gname, (r.logFC, -np.log10(r["adj.P.Val"])), fontsize=11, fontstyle="italic", fontweight="bold" if r.sig else "normal", color=c, xytext=off, textcoords="offset points", arrowprops=dict(arrowstyle="-", color="#888", lw=0.6))
    ax.axvline(-0.5, ls="--", color="#999", lw=.9); ax.axvline(0.5, ls="--", color="#999", lw=.9); ax.axhline(-np.log10(0.05), ls="--", color="#999", lw=.9)
    ax.text(0.98, 0.04, "q=0.05", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="#999", style="italic")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#D73027"), plt.Line2D([], [], marker="o", ls="", color="#2166AC")], labels=[f"Up ({nu})", f"Down ({nd})"], fontsize=12, loc="upper left", frameon=True)
    ax.set_xlabel("log₂FC (PC6-High / SMA-Low vs others), cancer-type-adjusted", fontsize=LS - 2); ax.set_ylabel("−log₁₀(q)  (FDR-adjusted)", fontsize=LS); ax.tick_params(labelsize=TS)
    ax.set_title(f"DEG: PC6-High/SMA-Low vs All Others\n(limma ~ group + cancer type; n = {nu + nd})", fontsize=TIS, fontweight="bold", pad=12); letter_at(ax, letter)
    ax.text(0.98, 0.10, "FDR < 0.05, |log₂FC| ≥ 0.5\ngrey labels: below the DEG threshold", transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="#555")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    NUM.append(dict(panel="4D", item="DEG total/up/down", value=f"{nu + nd}/{nu}/{nd}"))


def panel_violins(ax, letter="A"):
    MARK = [("LDHA", "Glycolysis"), ("SLC16A3", "MCT4"), ("HAVCR2", "T Exhaustion"), ("FOXP3", "Treg"), ("MSTN", "Cachexia")]; base = np.arange(len(MARK)) * 5.5
    for mi, (gene, cat) in enumerate(MARK):
        adj = wz(expr.loc[gene]); p = kruskal(*[adj[grp == q].values for q in G4]).pvalue; pf_ = f_group_after_ct(expr.loc[gene])
        for gi, q in enumerate(G4):
            vals = adj[grp == q].values; pos = base[mi] + gi; vp = ax.violinplot([vals], positions=[pos], widths=0.85, showmedians=True, showextrema=False)
            for b in vp["bodies"]: b.set_facecolor(COL[q]); b.set_alpha(0.85 if q == WORST else 0.5); b.set_edgecolor("none")
            vp["cmedians"].set_color("#222"); vp["cmedians"].set_linewidth(1.3)
        ax.text(base[mi] + 1.5, 3.4, f"{star(p)}  (KW {fe(p)})", ha="center", fontsize=11, color="#c0392b" if p < 0.05 else "#777", fontweight="bold")
        ax.text(base[mi] + 1.5, -3.95, f"$\\it{{{gene}}}$\n({cat})", ha="center", fontsize=12)
        NUM.append(dict(panel="S-A", item=f"{gene} KW p / F p", value=f"{p:.2g} / {pf_:.2g}"))
    ax.set_xticks([]); ax.set_ylabel("Expression (within-cancer-type z-score)", fontsize=LS - 2); ax.set_ylim(-4.4, 4.3); ax.axhline(0, color="#bbb", lw=.7); ax.tick_params(labelsize=TS)
    ax.set_title("Representative marker expression by PC6 × SMA subgroup (cancer-type-adjusted)", fontsize=TIS, fontweight="bold", pad=12); letter_at(ax, letter)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[q]) for q in G4], labels=G4, fontsize=11, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.09))
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)


def panel_km_by_type(fig, subspec, letter="B"):
    panels = [(coh, c_, s) for coh in ["TCGA", "CPTAC-3"] for c_, s in data[coh].groupby("cancer_type") if s.OS.sum() >= 10]
    inner = GridSpecFromSubplotSpec(2, 4, subplot_spec=subspec, hspace=0.55, wspace=0.3); axes = [fig.add_subplot(inner[i // 4, i % 4]) for i in range(8)]
    for ax, (coh, c_, s) in zip(axes, panels):
        for gname in G4:
            sub = s[s.group == gname]
            if len(sub) == 0: continue
            KaplanMeierFitter().fit(sub.OS_months, sub.OS, label=f"{SHORT[gname]} (n={len(sub)})").plot_survival_function(ax=ax, color=COL[gname], lw=1.9, ci_show=False, show_censors=True, censor_styles={"ms": 4, "marker": "+"})
        ng = s.group.nunique(); p = multivariate_logrank_test(s.OS_months, s.group, s.OS).p_value if ng >= 2 else np.nan
        ax.set_title(f"{coh}: {c_}  (n={len(s)}, events={int(s.OS.sum())})\nlog-rank {fp(p)}", fontsize=11); ax.set_ylim(0, 1.02); ax.set_xlabel("Months", fontsize=10); ax.set_ylabel("Overall survival", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower left", frameon=False); ax.tick_params(labelsize=9)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        NUM.append(dict(panel="S-B", cohort=coh, item=f"{c_} log-rank p", value=p))
    for ax in axes[len(panels):]: ax.axis("off")
    axes[0].text(-0.28, 1.28, letter, transform=axes[0].transAxes, fontsize=PS, fontweight="bold")
    axes[0].text(-0.12, 1.28, "Kaplan–Meier curves within each cancer type (cancer types with ≥ 10 deaths)", transform=axes[0].transAxes, fontsize=TIS, fontweight="bold", va="center")


def panel_sig_by_type(ax, fig, letter="C"):
    allc = pd.concat([PTd, pd.Series({k: v[0] for k, v in EFF.items()}, name="All types\n(stratified)")], axis=1)
    im = ax.imshow(np.ma.masked_invalid(allc.values), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto"); ax.set_facecolor("#e6e6e6")
    ax.set_xticks(range(allc.shape[1])); ax.set_xticklabels([f"{t}\nn={ntype[t]}\nworst {nworst[t]}" for t in TYPES] + [allc.columns[-1]], fontsize=11)
    ax.get_xticklabels()[-1].set_fontweight("bold"); ax.set_yticks(range(len(allc))); ax.set_yticklabels(allc.index, fontsize=12.5)
    for (yi, xi), val in np.ndenumerate(allc.values):
        if not np.isnan(val): ax.text(xi, yi, f"{val:+.2f}" + ("" if xi < len(TYPES) else f"\n{star(EFF[allc.index[yi]][3])}"), ha="center", va="center", fontsize=10.5, color="white" if abs(val) > 0.9 else "black", fontweight="normal" if xi < len(TYPES) else "bold")
        else: ax.text(xi, yi, "n<3", ha="center", va="center", fontsize=9.5, color="#777")
    ax.axvline(len(TYPES) - 0.5, color="#222", lw=1.3); cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("PC6-High / SMA-Low minus other three groups (within-type z)", fontsize=11); cb.ax.tick_params(labelsize=10)
    ax.set_title("Signature differences within each cancer type (TCGA)\nlast column: cancer-type-stratified estimate", fontsize=TIS - 1, fontweight="bold", pad=12); letter_at(ax, letter, x=-0.22)


def panel_deg_counts(ax, letter="D"):
    s_ = L[L.sig]; nu, nd = int((s_.logFC > 0).sum()), int((s_.logFC < 0).sum()); ov = len(set(s_.gene) & set(Lu[Lu.sig].gene))
    bars = [("Submitted\n(Wilcoxon, pooled)", 354, 470), ("limma, pooled\n(no cancer type)", int((Lu.sig & (Lu.logFC > 0)).sum()), int((Lu.sig & (Lu.logFC < 0)).sum())), ("limma + cancer type\n(adjusted; Fig. 4D)", nu, nd)]
    x = np.arange(3); ax.bar(x - 0.18, [b[1] for b in bars], 0.34, color="#D73027", label="up in PC6-High / SMA-Low"); ax.bar(x + 0.18, [b[2] for b in bars], 0.34, color="#2166AC", label="down")
    for i, b in enumerate(bars):
        ax.text(x[i] - 0.18, b[1] + 8, str(b[1]), ha="center", fontsize=11); ax.text(x[i] + 0.18, b[2] + 8, str(b[2]), ha="center", fontsize=11); ax.text(x[i], max(b[1], b[2]) + 75, f"total {b[1] + b[2]:,}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=11.5); ax.set_ylabel("DEGs (FDR < 0.05, |log₂FC| ≥ 0.5)", fontsize=LS - 3); ax.set_ylim(0, 870); ax.tick_params(labelsize=TS)
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    ax.set_title("DEG counts:\npooled vs cancer-type-adjusted", fontsize=TIS - 1, fontweight="bold", pad=12); letter_at(ax, letter, x=-0.20)
    ax.text(0.03, 0.97, f"{ov} of the {nu + nd} adjusted DEGs\nare also pooled-limma DEGs", transform=ax.transAxes, ha="left", va="top", fontsize=11, color="#444", style="italic")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    NUM.append(dict(panel="S-D", item="adjusted DEGs also pooled-limma DEGs", value=ov))


# ═════════════════════════════ main Figure 4 ═════════════════════════════
fig = plt.figure(figsize=(21, 18)); fig.patch.set_facecolor("white")
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.40, left=0.07, right=0.97, top=0.90, bottom=0.05, height_ratios=[1.15, 1.0], width_ratios=[1.0, 1.0])
panel_km(fig.add_subplot(gs[0, 0]), "A"); panel_forest(fig.add_subplot(gs[0, 1]), "B"); panel_heatmap(fig.add_subplot(gs[1, 0]), fig, "C"); panel_volcano(fig.add_subplot(gs[1, 1]), "D")
fig.suptitle("Joint Effects of PC6 and Body Composition", fontsize=23, fontweight="bold", y=0.985)
save(fig, "Figure4_final_layout")

# ═════════════════════════════ companion Supplementary figure ═════════════════════════════
fig = plt.figure(figsize=(21, 24)); fig.patch.set_facecolor("white")
gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35, left=0.07, right=0.97, top=0.95, bottom=0.04, height_ratios=[0.95, 1.75, 1.25], width_ratios=[1.45, 1.0])
panel_violins(fig.add_subplot(gs[0, :]), "A"); panel_km_by_type(fig, gs[1, :], "B"); panel_sig_by_type(fig.add_subplot(gs[2, 0]), fig, "C"); panel_deg_counts(fig.add_subplot(gs[2, 1]), "D")
save(fig, "SuppFig_Fig4_companion_stratified")
pd.DataFrame(NUM).to_csv(os.path.join(OUT, "numbers_final_layout.csv"), index=False, encoding="utf-8-sig"); print(pd.DataFrame(NUM).to_string())
