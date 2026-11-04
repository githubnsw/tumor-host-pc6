"""Fig 4 — every panel cancer-type-stratified, keeping the SUBMITTED group definition (PC6 x SMA by cohort-wide medians).
Drawn as separate, detailed panels (A1-A3, B1-B2, C1-C2, D1-D2) plus an overview sheet so the layout can be chosen afterwards.
Inputs: final_v2/inputs (clinical + PCs; TCGA identical to the submission), r2_8_stratified/degA (TCGA log2 expression matrix and
limma ~ group + cancer_type results for the cohort-wide groups). Notebooks untouched; new files only.
Stratification used:  survival -> stratified log-rank, Cox with strata = cancer type (age, sex, stage);  DEG -> limma with cancer type
as covariate;  signatures / markers -> within-cancer-type z-scores, Kruskal-Wallis on those + F-test for group after cancer type."""
import os, warnings, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.stats import kruskal, chi2, t as tdist
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); R1 = os.path.dirname(HERE)
FIN = os.path.join(R1, "final_v2", "inputs"); DEG = os.path.join(R1, "r2_8_stratified", "degA")
OUT = os.path.join(HERE, "panels"); os.makedirs(OUT, exist_ok=True)
G4 = ["PC6-Low / SMA-High", "PC6-Low / SMA-Low", "PC6-High / SMA-High", "PC6-High / SMA-Low"]; WORST = G4[3]
COL = {G4[0]: "#2166AC", G4[1]: "#74ADD1", G4[2]: "#F46D43", G4[3]: "#D73027"}
SHORT = {G4[0]: "PC6-L / SMA-H", G4[1]: "PC6-L / SMA-L", G4[2]: "PC6-H / SMA-H", G4[3]: "PC6-H / SMA-L"}
star = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
fp = lambda p: "p<0.001" if p < 0.001 else f"p={p:.3f}"
fe = lambda p: f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"
NUM = []


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=160, bbox_inches="tight", facecolor="white")
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
    d = d.dropna(subset=["PC6_z", "SMA_zscore", "OS_months", "OS"]).copy()          # same filter as the Fig 4 cell
    pm, sm = d.PC6_z.median(), d.SMA_zscore.median()                                  # cohort-wide medians, as submitted
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

# ───────────────────────── A1. pooled KM, TCGA, pooled + stratified log-rank ─────────────────────────
d = data["TCGA"]; fig, ax = plt.subplots(figsize=(8.2, 7.4)); kmfs = []
for g in G4:
    sub = d[d.group == g]; k = KaplanMeierFitter().fit(sub.OS_months, sub.OS, label=f"{SHORT[g]} (n={len(sub)})")
    k.plot_survival_function(ax=ax, color=COL[g], lw=2.4, ci_show=True, ci_alpha=0.08, show_censors=True, censor_styles={"ms": 5, "marker": "+"}); kmfs.append(k)
lrp = multivariate_logrank_test(d.OS_months, d.group, d.OS).p_value; st_, lrs = stratified_logrank(d)
ax.text(0.98, 0.04, f"Log-rank p = {fe(lrp)}\nStratified by cancer type: chi2 = {st_:.1f} (df 3), p = {fe(lrs)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF8E1", ec="#F39C12", lw=1.3))
ax.set_xlabel("Time (months)"); ax.set_ylabel("Overall survival probability"); ax.set_ylim(0, 1.02); ax.legend(fontsize=9.5, loc="upper right", frameon=True)
ax.set_title("A1  PC6 x SMA subgroups (cohort-wide medians, as submitted): TCGA overall survival", loc="left", fontsize=11.5, fontweight="bold")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
add_at_risk_counts(*kmfs, ax=ax, rows_to_show=["At risk"], fontsize=8.5); fig.subplots_adjust(bottom=0.32)
save(fig, "A1_KM_pooled_TCGA_stratified_p")
NUM += [dict(panel="A1", cohort="TCGA", item="pooled log-rank p", value=lrp), dict(panel="A1", cohort="TCGA", item="stratified log-rank p", value=lrs), dict(panel="A1", cohort="TCGA", item="stratified log-rank chi2", value=st_)]
for g in G4: NUM.append(dict(panel="A1", cohort="TCGA", item=f"n {g}", value=int((d.group == g).sum())))

# ───────────────────────── A2. stratified adjusted Cox forest, TCGA + CPTAC-3 ─────────────────────────
fig, ax = plt.subplots(figsize=(9.6, 6.4)); y = 0; ticks, labels, bold = [], [], []
for coh in ["TCGA", "CPTAC-3"]:
    d = data[coh]; hrs, n, ev, nref, adj = cox_groups(d); st_, lp = stratified_logrank(d)
    ticks.append(y); labels.append(f"{coh}  (Cox n = {n}, events = {ev})\nstratified log-rank {fp(lp)} (KM set n = {len(d)})"); bold.append(True); y -= 1.25
    ticks.append(y); labels.append(f"{G4[0]} (reference, n = {nref})"); bold.append(False); ax.plot(1, y, "s", color=COL[G4[0]], ms=7); y -= 1
    NUM += [dict(panel="A2", cohort=coh, item="stratified log-rank p (KM set)", value=lp, n=len(d)), dict(panel="A2", cohort=coh, item="Cox n / events / ref n", value=f"{n} / {ev} / {nref}"), dict(panel="A2", cohort=coh, item="adjusted for", value="+".join(adj))]
    for r in hrs.itertuples():
        ax.errorbar(r.HR, y, xerr=[[r.HR - r.lo], [r.hi - r.HR]], fmt="o", color=COL[r.group], ms=7, capsize=3.5, lw=1.8)
        ticks.append(y); labels.append(f"{r.group} (n = {r.n})"); bold.append(False)
        ax.text(13, y, f"{r.HR:.2f} ({r.lo:.2f}-{r.hi:.2f})   {fp(r.p)}  {star(r.p)}", fontsize=9.3, va="center", ha="left", family="monospace")
        NUM.append(dict(panel="A2", cohort=coh, item=f"{r.group} vs ref, stratified adjusted HR", value=f"{r.HR:.2f} ({r.lo:.2f}-{r.hi:.2f}) {fp(r.p)}", n=r.n)); y -= 1
    y -= 0.9
ax.axvline(1, color="#999", lw=0.9, ls="--"); ax.set_xscale("log"); ax.set_xlim(0.45, 120); ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5", "1", "2", "4", "8"]); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=9.3)
for lab, b in zip(ax.get_yticklabels(), bold): lab.set_fontweight("bold" if b else "normal")
ax.tick_params(axis="y", length=0); ax.set_ylim(y + 0.5, 0.9); ax.spines["bottom"].set_bounds(0.45, 10); ax.tick_params(axis="x", which="minor", bottom=False)
ax.set_xlabel("Hazard ratio vs PC6-Low / SMA-High (95% CI)\nCox stratified by cancer type, adjusted for age, sex, stage", fontsize=9.8)
ax.set_title("A2  Cancer-type-stratified Cox, same four subgroups (cohort-wide medians)", loc="left", fontsize=11.5, fontweight="bold")
for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
save(fig, "A2_stratified_Cox_forest_TCGA_CPTAC")

# ───────────────────────── A3. KM by cancer type (cohort-wide groups), types with >= 10 deaths ─────────────────────────
panels = [(coh, ct, s) for coh in ["TCGA", "CPTAC-3"] for ct, s in data[coh].groupby("cancer_type") if s.OS.sum() >= 10]
ncol = 4; nrow = math.ceil(len(panels) / ncol); fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow)); axes = np.array(axes).reshape(-1)
comp = []
for ax, (coh, ct, s) in zip(axes, panels):
    for g in G4:
        sub = s[s.group == g]
        if len(sub) == 0: continue
        KaplanMeierFitter().fit(sub.OS_months, sub.OS, label=f"{SHORT[g]} (n={len(sub)})").plot_survival_function(ax=ax, color=COL[g], lw=1.9, ci_show=False, show_censors=True, censor_styles={"ms": 4, "marker": "+"})
        comp.append(dict(cohort=coh, cancer_type=ct, group=g, n=len(sub), events=int(sub.OS.sum())))
    ng = s.group.nunique(); p = multivariate_logrank_test(s.OS_months, s.group, s.OS).p_value if ng >= 2 else np.nan
    ax.set_title(f"{coh}: {ct}  (n={len(s)}, events={int(s.OS.sum())})\nlog-rank {fp(p) if ng >= 2 else 'n/a'};  groups present: {ng}/4", fontsize=9.5)
    ax.set_ylim(0, 1.02); ax.set_xlabel("Months", fontsize=9); ax.set_ylabel("Overall survival", fontsize=9); ax.legend(fontsize=7.2, loc="lower left", frameon=False); ax.tick_params(labelsize=8)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    NUM.append(dict(panel="A3", cohort=coh, item=f"{ct} log-rank p (within type)", value=p, n=len(s)))
for ax in axes[len(panels):]: ax.axis("off")
fig.suptitle("A3  PC6 x SMA subgroups within each cancer type (cohort-wide medians, as submitted; cancer types with >= 10 deaths)", fontsize=12, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96)); save(fig, "A3_KM_by_cancer_type_cohortwide_groups"); pd.DataFrame(comp).to_csv(os.path.join(HERE, "A3_group_composition_by_cancer_type.csv"), index=False)

# ───────────────────────── B1. cancer-type-adjusted DEG volcano (limma ~ group + cancer type; cohort-wide groups) ─────────────────────────
L = pd.read_csv(os.path.join(DEG, "limma_cohortwide_adj_cancer_type.csv")); Lu = pd.read_csv(os.path.join(DEG, "limma_cohortwide_unadjusted.csv"))
s_ = L[L.sig]; nu, nd = int((s_.logFC > 0).sum()), int((s_.logFC < 0).sum()); ntot = nu + nd
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(L.logFC, -np.log10(L["adj.P.Val"]), s=5, color="#c8c8c8", linewidths=0)
ax.scatter(s_.logFC, -np.log10(s_["adj.P.Val"]), s=8, c=np.where(s_.logFC > 0, "#D73027", "#2166AC"), linewidths=0)
ANN = ["ALDOA", "SLC16A3", "PKM", "NDRG1", "CA9", "VEGFA", "LDHA", "FOXP3", "HAVCR2", "CD8A", "GZMB", "CXCL9", "CD163", "MRC1", "TGFB1", "MSTN"]
Ls = L.set_index("gene")
for gname in ANN:
    if gname in Ls.index:
        r = Ls.loc[gname]; c = ("#D73027" if r.logFC > 0 else "#2166AC") if r.sig else "#555"
        ax.annotate(gname, (r.logFC, -np.log10(r["adj.P.Val"])), fontsize=8.5, fontstyle="italic", color=c, xytext=(5, 4), textcoords="offset points", fontweight="bold" if r.sig else "normal")
        ax.plot(r.logFC, -np.log10(r["adj.P.Val"]), "o", ms=5, mfc="none", mec=c, mew=1.0)
        NUM.append(dict(panel="B1", cohort="TCGA", item=f"{gname} logFC / FDR / DEG (adjusted)", value=f"{r.logFC:+.2f} / {r['adj.P.Val']:.2g} / {bool(r.sig)}"))
ax.axvline(-0.5, ls="--", color="#999", lw=.8); ax.axvline(0.5, ls="--", color="#999", lw=.8); ax.axhline(-np.log10(0.05), ls="--", color="#999", lw=.8)
ax.text(0.02, 0.98, f"{ntot:,} DEGs (FDR < 0.05, |log2FC| >= 0.5)\nup {nu:,}   down {nd:,}\nlimma ~ group + cancer type (TCGA, n = 495)", transform=ax.transAxes, va="top", fontsize=9.8, bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
ax.set_xlabel("log2 fold change, PC6-High / SMA-Low vs other three groups (cancer-type-adjusted)"); ax.set_ylabel("-log10 (FDR)")
ax.set_title("B1  Differential expression, cancer type as covariate (cohort-wide groups)", loc="left", fontsize=11.5, fontweight="bold")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
save(fig, "B1_volcano_limma_adj_cancer_type"); NUM += [dict(panel="B1", cohort="TCGA", item="DEG total/up/down (adjusted)", value=f"{ntot}/{nu}/{nd}")]

# ───────────────────────── B2. DEG counts: submitted vs limma pooled vs limma + cancer type; overlap ─────────────────────────
su, sp_ = set(s_.gene), set(Lu[Lu.sig].gene); ov = len(su & sp_)
bars = [("Submitted\n(Wilcoxon, pooled)", 354, 470), ("limma, pooled\n(no cancer type)", int((Lu.sig & (Lu.logFC > 0)).sum()), int((Lu.sig & (Lu.logFC < 0)).sum())), ("limma + cancer type\n(adjusted)", nu, nd)]
fig, ax = plt.subplots(figsize=(6.4, 5.2)); x = np.arange(3)
ax.bar(x - 0.18, [b[1] for b in bars], 0.34, color="#D73027", label="up in PC6-High / SMA-Low"); ax.bar(x + 0.18, [b[2] for b in bars], 0.34, color="#2166AC", label="down")
for i, b in enumerate(bars):
    ax.text(x[i] - 0.18, b[1] + 8, str(b[1]), ha="center", fontsize=9); ax.text(x[i] + 0.18, b[2] + 8, str(b[2]), ha="center", fontsize=9); ax.text(x[i], max(b[1], b[2]) + 70, f"total {b[1] + b[2]:,}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=9.3); ax.set_ylabel("DEGs (FDR < 0.05, |log2FC| >= 0.5)"); ax.set_ylim(0, max(max(b[1], b[2]) for b in bars) * 1.3)
ax.legend(fontsize=8.5, frameon=False, loc="upper right")
ax.set_title(f"B2  DEG counts, PC6-High / SMA-Low vs others (TCGA)\n{ov} of the {ntot} adjusted DEGs are also pooled-limma DEGs", loc="left", fontsize=10.5, fontweight="bold")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
save(fig, "B2_DEG_counts_pooled_vs_adjusted"); NUM.append(dict(panel="B2", cohort="TCGA", item="adjusted DEGs also pooled-limma DEGs", value=ov))

# ───────────────────────── C/D. expression: within-cancer-type z; cohort-wide groups ─────────────────────────
expr = pd.read_csv(os.path.join(DEG, "expr_log2_tcga.csv"), index_col=0); g = pd.read_csv(os.path.join(DEG, "groups_tcga.csv")).set_index("pid").loc[expr.columns]
grp, ct = g["group4_cohortwide"], g["cancer_type"]; assert grp.value_counts().to_dict() == {G4[2]: 131, G4[1]: 130, G4[0]: 117, G4[3]: 117}, grp.value_counts().to_dict()
wz = lambda v: v.groupby(ct).transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))


def f_group_after_ct(v):
    dd = pd.DataFrame({"y": v.values, "group": grp.values, "ct": ct.values}); m = smf.ols("y ~ C(group) + C(ct)", data=dd).fit(); return float(anova_lm(m, typ=2).loc["C(group)", "PR(>F)"])


def worst_adj_effect(adjv):   # within-type z ~ worst + cancer type  -> stratified mean difference (worst - others) with 95% CI
    dd = pd.DataFrame({"y": adjv.values, "w": (grp == WORST).astype(float).values, "ct": ct.values}); m = smf.ols("y ~ w + C(ct)", data=dd).fit()
    return m.params["w"], m.conf_int().loc["w", 0], m.conf_int().loc["w", 1], m.pvalues["w"]


def welch_diff(a, b):
    a, b = a.values.astype(float), b.values.astype(float); dv = a.mean() - b.mean(); se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)); tq = tdist.ppf(0.975, max(min(len(a), len(b)) - 1, 1))
    return dv, dv - tq * se, dv + tq * se


SIGS = {"Glycolysis": "HK2,PKM,ALDOA,LDHA,GAPDH,ENO1", "Hypoxia": "CA9,NDRG1,VEGFA,HIF1A,ADM", "Lactate/MCT": "LDHB,SLC16A3,SLC16A1", "CD8 T cell": "CD8A,CD8B,GZMA,GZMB,PRF1",
        "T cell Exhaustion": "HAVCR2,PDCD1,LAG3,TOX", "Treg": "FOXP3,IL2RA,IKZF2", "M2 Macrophage": "MSR1,CD163,MRC1", "IFN-g Response": "STAT1,IRF1,CXCL9,CXCL10,IDO1",
        "Cachexia Signal": "MSTN,FBXO32,IL6,ACVR2B", "TGF-b": "TGFB1,TGFB2,SMAD2,SMAD3"}
SIGS = {k: [x for x in v.split(",") if x in expr.index] for k, v in SIGS.items()}
TYPES = sorted(ct.unique(), key=lambda t: -int((ct == t).sum())); nworst = {t: int(((ct == t) & (grp == WORST)).sum()) for t in TYPES}; ntype = {t: int((ct == t).sum()) for t in TYPES}
M, PK, PF, EFF, PT = {}, {}, {}, {}, {}
for name, genes in SIGS.items():
    score = expr.loc[genes].mean(axis=0); adj = wz(score)
    M[name] = adj.groupby(grp).mean().reindex(G4); PK[name] = kruskal(*[adj[grp == q].values for q in G4]).pvalue; PF[name] = f_group_after_ct(score); EFF[name] = worst_adj_effect(adj)
    PT[name] = {t: (adj[(ct == t) & (grp == WORST)].mean() - adj[(ct == t) & (grp != WORST)].mean()) if nworst[t] >= 3 else np.nan for t in TYPES}
M = pd.DataFrame(M).T[G4]; PTd = pd.DataFrame(PT).T[TYPES]
tab = pd.DataFrame({"kruskal_p_within_type_z": PK, "F_p_group_after_cancer_type": PF, "worst_vs_others_adj_diff": {k: v[0] for k, v in EFF.items()}, "adj_diff_lo": {k: v[1] for k, v in EFF.items()}, "adj_diff_hi": {k: v[2] for k, v in EFF.items()}, "adj_diff_p": {k: v[3] for k, v in EFF.items()}})
pd.concat([M.add_prefix("mean_wz["), tab], axis=1).to_csv(os.path.join(HERE, "C1_signature_within_type_z_by_group.csv"), encoding="utf-8-sig"); PTd.to_csv(os.path.join(HERE, "C2_signature_worst_minus_others_by_cancer_type.csv"), encoding="utf-8-sig")

# C1 heatmap
fig, ax = plt.subplots(figsize=(8.6, 7)); vmax = 1.0
im = ax.imshow(M.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([q.replace(" / ", "\n") for q in G4], fontsize=9.5)
for lab, q in zip(ax.get_xticklabels(), G4): lab.set_color(COL[q]); lab.set_fontweight("bold")
ax.set_yticks(range(len(M))); ax.set_yticklabels([f"{n}   {star(PK[n])}" for n in M.index], fontsize=10)
for (yi, xi), val in np.ndenumerate(M.values): ax.text(xi, yi, f"{val:+.2f}", ha="center", va="center", fontsize=9, color="white" if abs(val) > 0.6 else "black")
for yi, n in enumerate(M.index): ax.text(3.62, yi, f"KW {fe(PK[n])}\nF {fe(PF[n])}", va="center", ha="left", fontsize=7.6, color="#444")
ax.set_xlim(-0.5, 4.6); cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.16); cb.set_label("group mean of within-cancer-type z-scored signature score", fontsize=8.5)
ax.set_title("C1  Signature scores, cancer-type-adjusted (within-type z; cohort-wide groups)\nstars: Kruskal-Wallis on within-type z;  F: group after cancer type (OLS)", loc="left", fontsize=10.5, fontweight="bold")
save(fig, "C1_signature_heatmap_within_type_z")
for n in M.index: NUM += [dict(panel="C1", cohort="TCGA", item=f"{n} KW p (within-type z)", value=PK[n]), dict(panel="C1", cohort="TCGA", item=f"{n} F p (group | cancer type)", value=PF[n]), dict(panel="C1", cohort="TCGA", item=f"{n} worst-others adj. diff (z)", value=f"{EFF[n][0]:+.2f} ({EFF[n][1]:+.2f} to {EFF[n][2]:+.2f}) {fp(EFF[n][3])}")]

# C2 per-cancer-type heatmap: worst - others (within-type z), plus stratified estimate column
fig, ax = plt.subplots(figsize=(10.5, 7)); allc = pd.concat([PTd, pd.Series({k: v[0] for k, v in EFF.items()}, name="All types\n(stratified)")], axis=1)
im = ax.imshow(np.ma.masked_invalid(allc.values), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto"); ax.set_facecolor("#e6e6e6")
ax.set_xticks(range(allc.shape[1])); ax.set_xticklabels([f"{t}\nn={ntype[t]}\nworst {nworst[t]}" for t in TYPES] + [allc.columns[-1]], fontsize=8.4)
ax.get_xticklabels()[-1].set_fontweight("bold"); ax.set_yticks(range(len(allc))); ax.set_yticklabels(allc.index, fontsize=10)
for (yi, xi), val in np.ndenumerate(allc.values):
    if not np.isnan(val): ax.text(xi, yi, f"{val:+.2f}" + ("" if xi < len(TYPES) else f"\n{star(EFF[allc.index[yi]][3])}"), ha="center", va="center", fontsize=8 if xi < len(TYPES) else 8.5, color="white" if abs(val) > 0.9 else "black", fontweight="normal" if xi < len(TYPES) else "bold")
    else: ax.text(xi, yi, "n<3", ha="center", va="center", fontsize=7.5, color="#777")
ax.axvline(len(TYPES) - 0.5, color="#222", lw=1.2); cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("PC6-High / SMA-Low minus other three groups (within-type z units)", fontsize=8.5)
ax.set_title("C2  Signature difference, PC6-High / SMA-Low vs others, within each cancer type (TCGA)\nlast column: cancer-type-stratified estimate (OLS z ~ worst + cancer type) with significance", loc="left", fontsize=10.5, fontweight="bold")
save(fig, "C2_signature_worst_vs_others_by_cancer_type")

# D1 marker violins (5 markers as submitted), within-type z, cohort-wide groups
MARK = [("LDHA", "Glycolysis"), ("SLC16A3", "MCT4"), ("HAVCR2", "T Exhaustion"), ("FOXP3", "Treg"), ("MSTN", "Cachexia")]
fig, ax = plt.subplots(figsize=(10.5, 6)); base = np.arange(len(MARK)) * 5.5; DT = {}
for mi, (gene, cat) in enumerate(MARK):
    adj = wz(expr.loc[gene]); p = kruskal(*[adj[grp == q].values for q in G4]).pvalue; pf_ = f_group_after_ct(expr.loc[gene])
    DT[gene] = {t: (welch_diff(adj[(ct == t) & (grp == WORST)], adj[(ct == t) & (grp != WORST)]) if nworst[t] >= 3 else (np.nan, np.nan, np.nan)) for t in TYPES}
    DT[gene]["All"] = worst_adj_effect(adj)[:3]
    for gi, q in enumerate(G4):
        vals = adj[grp == q].values; pos = base[mi] + gi; vp = ax.violinplot([vals], positions=[pos], widths=0.85, showmedians=True, showextrema=False)
        for b in vp["bodies"]: b.set_facecolor(COL[q]); b.set_alpha(0.85 if q == WORST else 0.5); b.set_edgecolor("none")
        vp["cmedians"].set_color("#222"); vp["cmedians"].set_linewidth(1.2)
    ax.text(base[mi] + 1.5, 3.35, f"{star(p)}  KW {fe(p)}\nF {fe(pf_)}", ha="center", fontsize=8.5, color="#c0392b" if p < 0.05 else "#777", fontweight="bold")
    ax.text(base[mi] + 1.5, -3.9, f"$\\it{{{gene}}}$\n({cat})", ha="center", fontsize=9.5)
    NUM += [dict(panel="D1", cohort="TCGA", item=f"{gene} KW p (within-type z)", value=p), dict(panel="D1", cohort="TCGA", item=f"{gene} F p (group | cancer type)", value=pf_)]
ax.set_xticks([]); ax.set_ylabel("Expression (within-cancer-type z-score)"); ax.set_ylim(-4.3, 4.3); ax.axhline(0, color="#bbb", lw=.6)
ax.set_title("D1  Marker genes by PC6 x SMA subgroup, cancer-type-adjusted (within-type z; cohort-wide groups)", loc="left", fontsize=11, fontweight="bold")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[q]) for q in G4], labels=G4, fontsize=8.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2))
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
save(fig, "D1_marker_violins_within_type_z")

# D2 per-cancer-type worst-vs-others differences for the 5 markers, with stratified estimate
fig, axes = plt.subplots(1, len(MARK), figsize=(3.2 * len(MARK), 5.4), sharey=True); rowsD = []
for ax, (gene, cat) in zip(axes, MARK):
    ys = list(range(len(TYPES)))[::-1]
    for yv, t in zip(ys, TYPES):
        dv, lo, hi = DT[gene][t]
        if np.isnan(dv): ax.text(0, yv, "n<3", ha="center", va="center", fontsize=7.5, color="#999"); continue
        ax.errorbar(dv, yv, xerr=[[dv - lo], [hi - dv]], fmt="o", color="#555", ms=4.5, capsize=2.5, lw=1.1); rowsD.append(dict(gene=gene, cancer_type=t, diff=dv, lo=lo, hi=hi, n_worst=nworst[t], n_type=ntype[t]))
    dv, lo, hi = DT[gene]["All"]; ax.errorbar(dv, -1.2, xerr=[[dv - lo], [hi - dv]], fmt="D", color="#D73027", ms=7, capsize=3, lw=1.8); rowsD.append(dict(gene=gene, cancer_type="ALL_stratified", diff=dv, lo=lo, hi=hi))
    ax.axvline(0, ls="--", color="#999", lw=.8); ax.axhline(-0.4, color="#ccc", lw=.6); ax.set_yticks(ys + [-1.2]); ax.set_yticklabels([f"{t} (worst n={nworst[t]})" for t in TYPES] + ["All types (stratified)"], fontsize=8.3)
    labs = ax.get_yticklabels()
    if labs: labs[-1].set_fontweight("bold")
    ax.set_title(f"$\\it{{{gene}}}$ ({cat})", fontsize=10); ax.set_xlabel("worst - others (within-type z)", fontsize=8.5); ax.tick_params(labelsize=8)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.suptitle("D2  Marker difference, PC6-High / SMA-Low vs other groups, per cancer type (mean, 95% CI) and cancer-type-stratified (red)", fontsize=11, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "D2_marker_worst_vs_others_by_cancer_type"); pd.DataFrame(rowsD).to_csv(os.path.join(HERE, "D2_marker_worst_minus_others_by_cancer_type.csv"), index=False, encoding="utf-8-sig")

pd.DataFrame(NUM).to_csv(os.path.join(HERE, "fig4_all_stratified_numbers.csv"), index=False, encoding="utf-8-sig")

# ───────────────────────── overview sheet (PNG montage) ─────────────────────────
from PIL import Image, ImageDraw
names = [["A1_KM_pooled_TCGA_stratified_p", "A2_stratified_Cox_forest_TCGA_CPTAC", "B1_volcano_limma_adj_cancer_type"], ["A3_KM_by_cancer_type_cohortwide_groups", "B2_DEG_counts_pooled_vs_adjusted"], ["C1_signature_heatmap_within_type_z", "C2_signature_worst_vs_others_by_cancer_type"], ["D1_marker_violins_within_type_z", "D2_marker_worst_vs_others_by_cancer_type"]]
rows_img = []
for row in names:
    ims = [Image.open(os.path.join(OUT, n + ".png")).convert("RGB") for n in row]; H = 900
    ims = [im.resize((int(im.width * H / im.height), H)) for im in ims]; W = sum(im.width for im in ims) + 40 * (len(ims) + 1)
    canvas = Image.new("RGB", (W, H + 40), "white"); x = 40
    for im in ims: canvas.paste(im, (x, 20)); x += im.width + 40
    rows_img.append(canvas)
W = max(c.width for c in rows_img); sheet = Image.new("RGB", (W, sum(c.height for c in rows_img) + 120), "white"); y = 100
ImageDraw.Draw(sheet).text((40, 30), "Fig 4 - all panels cancer-type-stratified, submitted group definition (cohort-wide medians). Draft panels for layout decision (2026-11-04).", fill="black")
for c in rows_img: sheet.paste(c, (0, y)); y += c.height
sheet.save(os.path.join(HERE, "OVERVIEW_all_panels.png")); print("overview", sheet.size)
print(pd.DataFrame(NUM).to_string())
