"""
pc6_cox_spec.py — single source of truth for the PC6 / SMA Cox analyses (revision round 1).

The two notebook cells that produce manuscript Figure 2 ("Figure 4. Clinical Impact" in
Untitled3.ipynb) and Supplementary Figure 5A ("Figure 3. Selection of PC6" in Untitled1.ipynb)
each carried their own copy of the preprocessing + model code and drifted apart
(stage coding of '[Discrepancy]', ridge penalizer=0.1). Import from here instead:

    from pc6_cox_spec import load_cohort, preprocess_and_merge, fit_joint_model, screen_pcs

Unified spec
  * stage_bin   : explicit mapping. '[Discrepancy]', 'Unknown', '[Not Available]', NaN -> missing
  * Cox model   : lifelines CoxPHFitter, NO penalizer (Efron ties), strata = cancer_type,
                  covariates = age, sex_bin, stage_bin, SMA_zscore, AE_PC{k}_z
  * PC z-score  : within cohort over all merged patients (before exclusions) — as in the notebooks
  * exclusions  : OS_time <= 0, or missing stage / age / sex / SMA / OS
  * time unit   : months (days / 30.44)

Running this file directly writes the unified tables to  <BASE>/revision_r1/.
"""
import os
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2

warnings.filterwarnings("ignore")

def _resolve_base():
    """Data root: $BODYCOMP_BASE > ../data next to this file (zip package layout) > nsw's PC."""
    env = os.environ.get("BODYCOMP_BASE")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.normpath(os.path.join(here, "..", "data"))
    if os.path.exists(os.path.join(pkg, "TCGA_with_genes.csv")):
        return pkg
    return r"C:\Users\nsw\Desktop\body_composition"


BASE = _resolve_base()
AE_DIR = os.environ.get("BODYCOMP_AE_DIR") or os.path.join(BASE, "ae_v3_outputs")
if not os.path.exists(AE_DIR):  # zip package layout keeps the model under ../model
    AE_DIR = os.path.normpath(os.path.join(BASE, "..", "model", "ae_v3_outputs"))
# clinical tables can be overridden too (e.g. the STAR-TPM rebuild uses TCGA_with_genes_tpm.csv)
PATHS = {
    "TCGA":  (os.environ.get("BODYCOMP_TCGA_CLIN") or os.path.join(BASE, "TCGA_with_genes.csv"),
              os.path.join(AE_DIR, "ae_latent_pca10_all.csv")),
    "CPTAC": (os.environ.get("BODYCOMP_CPTAC_CLIN") or os.path.join(BASE, "External_with_genes.csv"),
              os.path.join(AE_DIR, "ae_latent_pca10_external.csv")),
}
COVARIATES = ["age", "sex_bin", "stage_bin", "SMA_zscore"]
N_PC = 10
MISSING_STAGE = {"", "NAN", "NONE", "UNKNOWN", "[DISCREPANCY]", "[NOT AVAILABLE]", "[NOT APPLICABLE]"}


# ----------------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------------
def format_stage_binary(x):
    """1 = advanced (III/IV), 0 = early (I/II), NaN = missing/unknown/discrepant."""
    s = str(x).upper().replace("STAGE", "").strip()
    if s in MISSING_STAGE:
        return np.nan
    if s.startswith(("III", "IV")):
        return 1.0
    if s.startswith(("I", "II")):
        return 0.0
    return np.nan


def load_cohort(cohort_name):
    clin_path, ae_path = PATHS[cohort_name]
    return pd.read_csv(clin_path), pd.read_csv(ae_path)


def preprocess_and_merge(clin_df, ae_df, cohort_name):
    """Same signature as the notebook function. Adds OS_time (months), sex_bin, stage_bin,
    AE_PC{k}_z and an `exclude_reason` column (empty string = usable)."""
    df = clin_df.copy()
    ae = ae_df.copy()

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["OS"] = pd.to_numeric(df["OS"], errors="coerce")
    df["OS_time"] = pd.to_numeric(df["OS_time"], errors="coerce") / 30.44
    df["SMA_zscore"] = pd.to_numeric(df["SMA_zscore"], errors="coerce")
    df["sex_bin"] = df["sex"].astype(str).str.upper().map({"MALE": 1.0, "FEMALE": 0.0})
    df["stage_bin"] = df["stage"].apply(format_stage_binary)

    if cohort_name == "TCGA":
        df["pid"] = df["patient_id"].astype(str).str[:12]
        ae["pid"] = ae["patient_id"].astype(str).str[:12]
    else:
        df["pid"] = df["patient_id"].astype(str)
        ae["pid"] = ae["patient_id"].astype(str)

    use_cols = ["pid"] + [f"AE_PC{i}" for i in range(1, N_PC + 1)]
    merged = df.merge(ae[use_cols], on="pid", how="inner")

    for i in range(1, N_PC + 1):
        col = f"AE_PC{i}"
        std = merged[col].std()
        merged[f"{col}_z"] = (merged[col] - merged[col].mean()) / std if std > 0 else np.nan

    reasons = []
    for _, r in merged.iterrows():
        why = []
        if pd.isna(r["OS_time"]) or pd.isna(r["OS"]):
            why.append("missing_OS")
        elif r["OS_time"] <= 0:
            why.append("OS_time<=0")
        if pd.isna(r["stage_bin"]):
            why.append(f"missing_stage({r['stage']})")
        if pd.isna(r["age"]):
            why.append("missing_age")
        if pd.isna(r["sex_bin"]):
            why.append("missing_sex")
        if pd.isna(r["SMA_zscore"]):
            why.append("missing_SMA")
        reasons.append("; ".join(why))
    merged["exclude_reason"] = reasons
    return merged


def analytic_subset(df, pc_col):
    """Rows usable for a Cox model on `pc_col` (+ the excluded rows, with reasons)."""
    cols = ["OS_time", "OS", "cancer_type"] + COVARIATES + [pc_col]
    ok = (df["exclude_reason"] == "") & df[pc_col].notna()
    return df.loc[ok, cols].copy(), df.loc[~ok]


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def fit_cox(d, pc_col, interaction=False):
    d = d.copy()
    if interaction:
        d["SMA_PC_interaction"] = d["SMA_zscore"] * d[pc_col]
    cph = CoxPHFitter()  # no penalizer
    cph.fit(d, duration_col="OS_time", event_col="OS", strata="cancer_type")
    return cph, d


def summarize(cph, variables, selected_pc=None):
    rows = []
    for v in variables:
        s = cph.summary.loc[v]
        label = "SMA z-score" if v == "SMA_zscore" else (selected_pc if v == f"AE_{selected_pc}_z" else v)
        rows.append({"variable": v, "label": label,
                     "hr": float(s["exp(coef)"]),
                     "lo": float(s["exp(coef) lower 95%"]),
                     "hi": float(s["exp(coef) upper 95%"]),
                     "p": float(s["p"])})
    return pd.DataFrame(rows)


def fit_joint_model(df, selected_pc="PC6"):
    """Drop-in replacement for the notebook function: returns (cph, analytic_df, result_df)."""
    pc_col = f"AE_{selected_pc}_z"
    tmp, _ = analytic_subset(df, pc_col)
    cph, tmp = fit_cox(tmp, pc_col)
    return cph, tmp, summarize(cph, ["SMA_zscore", pc_col], selected_pc)


def screen_pcs(df_t, df_e):
    """Drop-in replacement for the per-PC screening loop (columns PC, t_hr.. e_p)."""
    rows = []
    for i in range(1, N_PC + 1):
        pc_col = f"AE_PC{i}_z"
        row = {"PC": f"PC{i}"}
        for tag, df_ in [("t", df_t), ("e", df_e)]:
            d, _ = analytic_subset(df_, pc_col)
            cph, _ = fit_cox(d, pc_col)
            s = cph.summary.loc[pc_col]
            row[f"{tag}_n"] = len(d)
            row[f"{tag}_events"] = int(d["OS"].sum())
            row[f"{tag}_hr"] = s["exp(coef)"]
            row[f"{tag}_lo"] = s["exp(coef) lower 95%"]
            row[f"{tag}_hi"] = s["exp(coef) upper 95%"]
            row[f"{tag}_p"] = s["p"]
        row["concordant"] = (row["t_hr"] > 1) == (row["e_hr"] > 1)
        row["both_sig"] = (row["t_p"] < 0.05) and (row["e_p"] < 0.05)
        row["bilateral_score"] = -np.log10(row["t_p"]) - np.log10(row["e_p"])
        rows.append(row)
    return pd.DataFrame(rows)


def interaction_lr_test(df, selected_pc="PC6"):
    pc_col = f"AE_{selected_pc}_z"
    d, _ = analytic_subset(df, pc_col)
    cph0, _ = fit_cox(d, pc_col)
    cph1, _ = fit_cox(d, pc_col, interaction=True)
    lr = 2 * (cph1.log_likelihood_ - cph0.log_likelihood_)
    return cph1, lr, chi2.sf(lr, 1)


def breslow_crosscheck(d, pc_col):
    """statsmodels PHReg (Breslow ties) HR for the PC — the method named in the submitted Methods."""
    from statsmodels.duration.hazard_regression import PHReg
    cols = ["SMA_zscore", pc_col] + ["age", "sex_bin", "stage_bin"]
    r = PHReg(d["OS_time"], d[cols], status=d["OS"], strata=d["cancer_type"], ties="breslow").fit()
    i = cols.index(pc_col)
    return float(np.exp(r.params[i])), float(r.pvalues[i])


# ----------------------------------------------------------------------------
# main: write unified tables
# ----------------------------------------------------------------------------
SUBMITTED = [  # values quoted in the submitted manuscript (2026-06-04)
    ("screening", "TCGA", "PC6", 1.191, 1.015, 1.398, 0.032, 489, 188),
    ("screening", "CPTAC", "PC6", 1.404, 1.078, 1.828, 0.012, 178, 72),
    ("screening", "TCGA", "PC5", 0.803, np.nan, np.nan, 0.0009, 489, 188),
    ("screening", "TCGA", "PC8", 0.813, np.nan, np.nan, 0.010, 489, 188),
    ("joint", "TCGA", "SMA z-score", 0.76, 0.66, 0.89, 0.0004, 491, 190),
    ("joint", "TCGA", "PC6", 1.17, 1.00, 1.37, 0.054, 491, 190),
    ("joint", "CPTAC", "SMA z-score", 0.84, 0.64, 1.10, 0.205, 178, 72),
    ("joint", "CPTAC", "PC6", 1.40, 1.08, 1.83, 0.012, 178, 72),
]


def main(outdir=None, selected_pc="PC6"):
    outdir = outdir or os.path.join(BASE, "revision_r1")
    os.makedirs(outdir, exist_ok=True)
    pc_col = f"AE_{selected_pc}_z"
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    data = {c: preprocess_and_merge(*load_cohort(c), c) for c in ("TCGA", "CPTAC")}

    # 1) exclusions
    excl = []
    for c, df in data.items():
        _, ex = analytic_subset(df, pc_col)
        ex = ex[["patient_id", "cancer_type", "stage", "OS", "OS_time", "exclude_reason"]].copy()
        ex.insert(0, "cohort", c)
        excl.append(ex)
    excl = pd.concat(excl)
    excl.to_csv(os.path.join(outdir, "excluded_patients.csv"), index=False)

    say("=" * 78); say("1) Analytic samples"); say("=" * 78)
    for c, df in data.items():
        d, ex = analytic_subset(df, pc_col)
        say(f"  {c:6s} merged={len(df)}  analytic n={len(d)}  events={int(d['OS'].sum())}  excluded={len(ex)}")
    say("  excluded patients:")
    for _, r in excl.iterrows():
        say(f"    {r.cohort:6s} {r.patient_id:14s} {r.cancer_type:12s} stage={str(r.stage):14s} OS={r.OS} t={r.OS_time:.1f}m  -> {r.exclude_reason}")

    # 2) screening
    scr = screen_pcs(data["TCGA"], data["CPTAC"])
    scr.to_csv(os.path.join(outdir, "screening_all_pcs.csv"), index=False)
    say(); say("=" * 78); say("2) Per-PC screening (unified spec)"); say("=" * 78)
    say(f"  {'PC':5s} {'TCGA HR (95% CI)':26s} {'p':>8s}   {'CPTAC HR (95% CI)':26s} {'p':>8s}   both_sig concordant")
    for _, r in scr.iterrows():
        say(f"  {r.PC:5s} {r.t_hr:.3f} ({r.t_lo:.3f}-{r.t_hi:.3f})       {r.t_p:8.4f}   "
            f"{r.e_hr:.3f} ({r.e_lo:.3f}-{r.e_hi:.3f})       {r.e_p:8.4f}   {str(r.both_sig):8s} {r.concordant}")
    sel = scr[scr.both_sig & scr.concordant].sort_values("bilateral_score", ascending=False)
    say(f"  -> components significant & concordant in both cohorts: {sel.PC.tolist() or 'none'}")

    # 3) joint + interaction
    rows, lr_rows = [], []
    say(); say("=" * 78); say("3) Joint model SMA + PC6 (adj. age, sex, stage | strata cancer type)"); say("=" * 78)
    for c, df in data.items():
        cph, d, res = fit_joint_model(df, selected_pc)
        b_hr, b_p = breslow_crosscheck(d, pc_col)
        n, ev = len(d), int(d["OS"].sum())
        say(f"  [{c}] n={n}, events={ev}")
        for _, r in res.iterrows():
            say(f"    {r.label:12s} HR={r.hr:.3f} ({r.lo:.3f}-{r.hi:.3f})  p={r.p:.4f}")
            rows.append({"cohort": c, "model": "joint", "variable": r.variable, "label": r.label,
                         "HR": r.hr, "CI_low": r.lo, "CI_high": r.hi, "p_value": r.p, "n": n, "events": ev})
        say(f"    (cross-check PC6 via statsmodels Breslow: HR={b_hr:.3f}, p={b_p:.4f})")
        cph1, lr, lr_p = interaction_lr_test(df, selected_pc)
        res1 = summarize(cph1, ["SMA_zscore", pc_col, "SMA_PC_interaction"], selected_pc)
        for _, r in res1.iterrows():
            rows.append({"cohort": c, "model": "interaction", "variable": r.variable, "label": r.label,
                         "HR": r.hr, "CI_low": r.lo, "CI_high": r.hi, "p_value": r.p, "n": n, "events": ev})
        lr_rows.append({"cohort": c, "LR_stat": lr, "LR_p_value": lr_p})
        say(f"    interaction SMA x PC6: HR={res1.iloc[2].hr:.3f} p={res1.iloc[2].p:.4f}; LR test p={lr_p:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "joint_model_pc6.csv"), index=False)
    pd.DataFrame(lr_rows).to_csv(os.path.join(outdir, "interaction_lr_test.csv"), index=False)

    # 4) submitted vs revised
    joint = pd.DataFrame(rows)
    cmp_rows = []
    for kind, c, label, hr, lo, hi, p, n, ev in SUBMITTED:
        if kind == "screening":
            r = scr[scr.PC == label].iloc[0]; tag = "t" if c == "TCGA" else "e"
            new = (r[f"{tag}_hr"], r[f"{tag}_lo"], r[f"{tag}_hi"], r[f"{tag}_p"], r[f"{tag}_n"], r[f"{tag}_events"])
        else:
            r = joint[(joint.cohort == c) & (joint.model == "joint") & (joint.label == label)].iloc[0]
            new = (r.HR, r.CI_low, r.CI_high, r.p_value, r.n, r.events)
        cmp_rows.append({"table": kind, "cohort": c, "variable": label,
                         "submitted_HR": hr, "submitted_CI": f"{lo:.2f}-{hi:.2f}" if not np.isnan(lo) else "",
                         "submitted_p": p, "submitted_n": n, "submitted_events": ev,
                         "revised_HR": round(new[0], 3), "revised_CI": f"{new[1]:.3f}-{new[2]:.3f}",
                         "revised_p": round(new[3], 4), "revised_n": int(new[4]), "revised_events": int(new[5])})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(os.path.join(outdir, "submitted_vs_revised.csv"), index=False)
    say(); say("=" * 78); say("4) Submitted manuscript vs unified spec"); say("=" * 78)
    say(f"  {'table':10s} {'cohort':6s} {'variable':12s} {'submitted':34s} {'revised':34s}")
    for _, r in cmp.iterrows():
        s_old = f"HR {r.submitted_HR} ({r.submitted_CI}) p={r.submitted_p} n={r.submitted_n}"
        s_new = f"HR {r.revised_HR} ({r.revised_CI}) p={r.revised_p} n={r.revised_n}"
        say(f"  {r.table:10s} {r.cohort:6s} {r.variable:12s} {s_old:34s} {s_new:34s}")

    with open(os.path.join(outdir, "SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    say(); say(f"written to {outdir}")


if __name__ == "__main__":
    main()
