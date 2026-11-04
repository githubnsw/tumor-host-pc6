"""
Supplementary Figure 7 (cancer-type sensitivity of the PC6–survival association), recomputed from data.

Replaces the submitted supp_figure_sensitivity_forest_v2.tiff, whose values were typed into the plotting cell
(Untitled3.ipynb, first cell) and do not match the cohort (see revision_r1/manuscript_number_audit.md).

Two specifications are produced:
  unified   - pc6_cox_spec: no penalizer, '[Discrepancy]'/Unknown stage -> missing, OS_time <= 0 excluded
  submitted - as in the submitted Fig 2 cell: lifelines penalizer=0.1, '[Discrepancy]' coded as early stage
Model (both): Cox for overall survival on PC6 (z-score, per 1 SD) adjusted for SMA z-score, age, sex, stage.
  Pooled rows are stratified by cancer type; per-cancer rows are fitted within that cancer type
  (covariates constant within a cancer type, e.g. sex in UCEC, are dropped).
  Cancer types with fewer than MIN_EVENTS deaths are reported as not estimable.
Heterogeneity: likelihood-ratio test of PC6 x cancer-type interaction terms (cancer types with >= MIN_EVENTS deaths).

Outputs (this folder): supp_fig7_<spec>.tiff / .png, supp_fig7_<spec>.csv, supp_fig7_vs_submitted.csv, SUMMARY.txt
"""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from lifelines import CoxPHFitter
from scipy.stats import chi2

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\nsw\body_com_paper_code")
import pc6_cox_spec as spec  # noqa: E402

PC = "AE_PC6_z"
COVS = ["SMA_zscore", "age", "sex_bin", "stage_bin"]
MIN_EVENTS = 5
ORDER = ["BLCA", "COAD", "KIRC", "KIRP", "LIHC", "READ", "STAD", "UCEC"]
EXCLUDED = ["KIRC", "READ"]
SPECS = {"unified": 0.0, "submitted": 0.1}

# values hard-coded in the submitted figure (Untitled3.ipynb first cell)
SUBMITTED = {"BLCA": (105, 49, 1.03, 0.92, 1.15, 0.635), "COAD": (21, 8, 1.03, 0.50, 2.10, 0.944),
             "KIRC": (171, 47, 1.07, 0.92, 1.24, 0.385), "KIRP": (22, 6, 0.95, 0.46, 1.94, 0.881),
             "LIHC": (72, 34, 1.38, 1.18, 1.61, 0.000), "STAD": (43, 28, 0.82, 0.62, 1.09, 0.173),
             "Overall (full)": (494, 192, 1.065, 1.002, 1.132, 0.042), "Overall (KIRC+READ excluded)": (320, 129, 1.068, 0.997, 1.143, 0.059)}


def stage_submitted(x):
    s = str(x).upper()
    if "III" in s or "IV" in s:
        return 1.0
    if "II" in s or "I" in s:
        return 0.0
    return np.nan


def analytic(name):
    clin, ae = spec.load_cohort("TCGA")
    df = spec.preprocess_and_merge(clin, ae, "TCGA")
    if name == "unified":
        d, _ = spec.analytic_subset(df, PC)
    else:
        df = df.copy()
        df["stage_bin"] = df["stage"].apply(stage_submitted)
        d = df[["OS_time", "OS", "cancer_type"] + COVS + [PC]].dropna()
        d = d[d["OS_time"] > 0]
    return d.reset_index(drop=True)


def fit(d, pen, strata):
    covs = [c for c in COVS if d[c].nunique() > 1]
    cols = ["OS_time", "OS", PC] + covs + (["cancer_type"] if strata else [])
    c = CoxPHFitter(penalizer=pen).fit(d[cols], "OS_time", "OS", strata="cancer_type" if strata else None)
    s = c.summary.loc[PC]
    return {"hr": s["exp(coef)"], "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"], "p": s["p"],
            "covariates": "+".join(covs)}


def heterogeneity(d, pen):
    types = [t for t in ORDER if t in set(d.cancer_type) and d.loc[d.cancer_type == t, "OS"].sum() >= MIN_EVENTS]
    sub = d[d.cancer_type.isin(types)].copy()
    base = ["OS_time", "OS", "cancer_type", PC] + COVS
    m0 = CoxPHFitter(penalizer=pen).fit(sub[base], "OS_time", "OS", strata="cancer_type")
    inter = []
    for t in types[1:]:
        col = f"PC6_x_{t}"; sub[col] = sub[PC] * (sub.cancer_type == t); inter.append(col)
    m1 = CoxPHFitter(penalizer=pen).fit(sub[base + inter], "OS_time", "OS", strata="cancer_type")
    lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
    return lr, len(inter), chi2.sf(lr, len(inter)), types


def panel_rows(d, pen, types):
    rows = []
    for t in types:
        g = d[d.cancer_type == t]
        if len(g) == 0:
            continue
        r = {"group": t, "n": len(g), "events": int(g.OS.sum())}
        if r["events"] < MIN_EVENTS:
            r.update(hr=np.nan, lo=np.nan, hi=np.nan, p=np.nan, note=f"not estimable (<{MIN_EVENTS} events)")
        else:
            try:
                r.update(fit(g, pen, strata=False), note="")
            except Exception as ex:
                r.update(hr=np.nan, lo=np.nan, hi=np.nan, p=np.nan, note=f"did not converge ({type(ex).__name__})")
        rows.append(r)
    g = d[d.cancer_type.isin(types)]
    rows.append({"group": "Pooled", "n": len(g), "events": int(g.OS.sum()), **fit(g, pen, strata=True), "note": "stratified by cancer type"})
    return rows


# ---------------------------------------------------------------- drawing
INK, INK2, MUTED = "#1A1A1A", "#4D4D4D", "#8C8C8C"
MARK = "#1F4E79"
SHADE = "#F3F4F6"
XMIN, XMAX = 0.2, 5.0


def fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def draw_panel(fig, slot, rows, title):
    n = len(rows)
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=slot, width_ratios=[1.55, 2.6, 2.05], wspace=0)
    axl, axm, axr = (fig.add_subplot(inner[i]) for i in range(3))
    ytop = n + 0.6
    ys = list(range(n, 1, -1)) + [0]  # cancer rows from top, pooled row at the bottom after a gap
    for ax in (axl, axm, axr):
        ax.set_ylim(-0.8, ytop + 0.4)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    for ax in (axl, axr):
        ax.set_xlim(0, 1); ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)
    axm.set_xscale("log"); axm.set_xlim(XMIN, XMAX)
    axm.set_xticks([0.25, 0.5, 1, 2, 4]); axm.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    axm.minorticks_off()
    axm.tick_params(axis="x", labelsize=9, colors=INK2, length=3)
    axm.spines["bottom"].set_color(MUTED)
    axm.set_xlabel("Hazard ratio per 1-SD increase in PC6 (log scale)", fontsize=9.5, color=INK2)
    axm.axvline(1, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)

    axl.text(0.03, ytop, "Cancer type", fontsize=9.5, fontweight="bold", color=INK, va="bottom")
    axl.text(0.97, ytop, "n (deaths)", fontsize=9.5, fontweight="bold", color=INK, va="bottom", ha="right")
    axr.text(0.06, ytop, "HR (95% CI)", fontsize=9.5, fontweight="bold", color=INK, va="bottom")
    axr.text(0.97, ytop, "p", fontsize=9.5, fontweight="bold", color=INK, va="bottom", ha="right")
    axm.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left", pad=22, x=-0.6)

    for i, (r, y) in enumerate(zip(rows, ys)):
        pooled = r["group"] == "Pooled"
        if i % 2 == 0 and not pooled:
            for ax in (axl, axm, axr):
                ax.axhspan(y - 0.45, y + 0.45, color=SHADE, zorder=0, lw=0)
        fw = "bold" if pooled else "normal"
        label = "Pooled" if pooled else r["group"]
        axl.text(0.03, y, label, fontsize=9.5, fontweight=fw, color=INK, va="center")
        axl.text(0.97, y, f"{r['n']} ({r['events']})", fontsize=9.5, fontweight=fw, color=INK2, va="center", ha="right")
        if np.isnan(r["hr"]):
            axr.text(0.06, y, r["note"], fontsize=8.5, color=MUTED, va="center", style="italic")
            continue
        lo, hi = max(r["lo"], XMIN * 1.02), min(r["hi"], XMAX / 1.02)
        axm.plot([lo, hi], [y, y], color=MARK, lw=1.6 if not pooled else 2.2, solid_capstyle="round", zorder=2)
        # arrowheads only where the CI runs past the axis limits
        if r["lo"] < XMIN:
            axm.annotate("", xy=(lo, y), xytext=(lo * 1.15, y), arrowprops=dict(arrowstyle="-|>", color=MARK, lw=1.2, mutation_scale=9), zorder=2)
        if r["hi"] > XMAX:
            axm.annotate("", xy=(hi, y), xytext=(hi / 1.15, y), arrowprops=dict(arrowstyle="-|>", color=MARK, lw=1.2, mutation_scale=9), zorder=2)
        axm.scatter([r["hr"]], [y], s=95 if pooled else 46, marker="D" if pooled else "s", color=MARK,
                    edgecolors="white", linewidths=1.2, zorder=3)
        axr.text(0.06, y, f"{r['hr']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})", fontsize=9.5, fontweight=fw, color=INK, va="center")
        axr.text(0.97, y, fmt_p(r["p"]), fontsize=9.5, fontweight="bold" if r["p"] < 0.05 else fw, color=INK, va="center", ha="right")
    for ax in (axl, axm, axr):
        ax.axhline(0.85, color=MUTED, lw=0.8)


def make_figure(name, full, excl, het_full, het_excl, out_base):
    fig = plt.figure(figsize=(17, 6.6), facecolor="white")
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.10, left=0.02, right=0.985, top=0.86, bottom=0.2)
    pf, pe = full[-1], excl[-1]
    draw_panel(fig, outer[0], full, f"A   Full cohort (n={pf['n']}, {pf['events']} deaths)")
    draw_panel(fig, outer[1], excl, f"B   KIRC and READ excluded (n={pe['n']}, {pe['events']} deaths)")
    spec_txt = ("no penalization; '[Discrepancy]'/unknown stage treated as missing" if name == "unified"
                else "ridge penalizer 0.1 and '[Discrepancy]' stage coded as early, as in the submitted Figure 2")
    foot = (f"Cox proportional hazards models for overall survival (TCGA); PC6 standardized within cohort; adjusted for SMA z-score, age, sex and stage; {spec_txt}.\n"
            f"Pooled rows stratified by cancer type; per-cancer rows fitted within each cancer type (covariates constant within a cancer type omitted). "
            f"Cancer types with <{MIN_EVENTS} deaths not estimated.\n"
            f"Heterogeneity of the PC6 effect across cancer types (likelihood-ratio test of PC6 × cancer type): "
            f"A, χ²={het_full[0]:.2f}, df={het_full[1]}, p={het_full[2]:.3f}; B, χ²={het_excl[0]:.2f}, df={het_excl[1]}, p={het_excl[2]:.3f}.")
    fig.text(0.02, 0.03, foot, fontsize=8.3, color=INK2, va="bottom", linespacing=1.5)
    fig.savefig(out_base + ".tiff", dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(out_base + ".png", dpi=130, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- main
lines = []
say = lambda s="": (print(s), lines.append(s))
all_rows = []
for name, pen in SPECS.items():
    d = analytic(name)
    full = panel_rows(d, pen, ORDER)
    excl = panel_rows(d, pen, [t for t in ORDER if t not in EXCLUDED])
    het_full, het_excl = heterogeneity(d, pen), heterogeneity(d[~d.cancer_type.isin(EXCLUDED)], pen)
    make_figure(name, full, excl, het_full, het_excl, os.path.join(HERE, f"supp_fig7_{name}"))
    tab = pd.DataFrame([{**r, "panel": "A_full"} for r in full] + [{**r, "panel": "B_excl_KIRC_READ"} for r in excl])
    tab.insert(0, "spec", name); tab.to_csv(os.path.join(HERE, f"supp_fig7_{name}.csv"), index=False)
    all_rows.append(tab)
    say("=" * 96); say(f"spec = {name} (penalizer {pen})  analytic n={len(d)}, deaths={int(d.OS.sum())}"); say("=" * 96)
    for panel, rows, het in (("A full cohort", full, het_full), ("B KIRC+READ excluded", excl, het_excl)):
        say(f"  {panel}")
        for r in rows:
            hr = "—" if np.isnan(r["hr"]) else f"{r['hr']:.2f} ({r['lo']:.2f}–{r['hi']:.2f}) {fmt_p(r['p'])}"
            say(f"    {r['group']:8s} n={r['n']:3d} deaths={r['events']:3d}  {hr:32s} {r['note']}")
        say(f"    heterogeneity LR chi2={het[0]:.2f} df={het[1]} p={het[2]:.3f}  (types: {', '.join(het[3])})")

# comparison with the hard-coded submitted values
uni = all_rows[0].set_index(["panel", "group"])
cmp = []
for k, (n, ev, hr, lo, hi, p) in SUBMITTED.items():
    key = ("A_full", "Pooled") if k == "Overall (full)" else (("B_excl_KIRC_READ", "Pooled") if k.startswith("Overall (KIRC") else ("A_full", k))
    r = uni.loc[key]
    cmp.append({"row": k, "submitted_n": n, "submitted_deaths": ev, "submitted_HR": hr, "submitted_CI": f"{lo}–{hi}", "submitted_p": p,
                "recomputed_n": int(r.n), "recomputed_deaths": int(r.events),
                "recomputed_HR": None if np.isnan(r.hr) else round(r.hr, 3),
                "recomputed_CI": "" if np.isnan(r.hr) else f"{r.lo:.2f}–{r.hi:.2f}",
                "recomputed_p": None if np.isnan(r.hr) else round(r.p, 4), "note": r.note})
cmp = pd.DataFrame(cmp); cmp.to_csv(os.path.join(HERE, "supp_fig7_vs_submitted.csv"), index=False)
say(); say("submitted (hard-coded) vs recomputed (unified spec; submitted HRs were not per-SD, so only n/deaths are directly comparable):")
say(cmp.to_string(index=False))
open(os.path.join(HERE, "SUMMARY.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
