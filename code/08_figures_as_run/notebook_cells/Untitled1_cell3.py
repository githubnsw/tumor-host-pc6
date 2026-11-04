import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight
from scipy.stats import chi2, norm

# ============================================================
# CONFIG
# ============================================================
BASE = r"C:\Users\nsw\Desktop\body_composition"

TCGA_PATH = os.path.join(BASE, "TCGA_with_genes.csv")
EXT_PATH  = os.path.join(BASE, "External_with_genes.csv")
AE_TCGA   = os.path.join(BASE, "ae_latent_pca10_all.csv")
AE_EXT    = os.path.join(BASE, "ae_latent_pca10_external.csv")

OUTDIR = os.path.join(BASE, "clinical_impact_pc6")
os.makedirs(OUTDIR, exist_ok=True)

TAU_MONTHS = 60.0
SELECTED_PC = "PC6"


# ============================================================
# HELPERS
# ============================================================
def format_stage_binary(x):
    s = str(x).upper()
    if ("III" in s) or ("IV" in s):
        return 1.0
    if ("II" in s) or ("I" in s):
        return 0.0
    return np.nan


def preprocess_and_merge(clin_df, ae_df, cohort_name):
    df = clin_df.copy()
    ae = ae_df.copy()

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["OS"] = pd.to_numeric(df["OS"], errors="coerce")
    df["OS_time_months"] = pd.to_numeric(df["OS_time"], errors="coerce") / 30.44
    df["SMA_zscore"] = pd.to_numeric(df["SMA_zscore"], errors="coerce")
    df["sex_bin"] = (df["sex"].astype(str).str.upper() == "MALE").astype(float)
    df["stage_bin"] = df["stage"].apply(format_stage_binary)

    if cohort_name == "TCGA":
        df["pid"] = df["patient_id"].astype(str).str[:12]
        ae["pid"] = ae["patient_id"].astype(str).str[:12]
    else:
        df["pid"] = df["patient_id"].astype(str)
        ae["pid"] = ae["patient_id"].astype(str)

    use_cols = ["pid"] + [f"AE_PC{i}" for i in range(1, 11)]
    merged = df.merge(ae[use_cols], on="pid", how="inner")

    for i in range(1, 11):
        col = f"AE_PC{i}"
        zcol = f"{col}_z"
        std = merged[col].std()
        if pd.isna(std) or std == 0:
            merged[zcol] = np.nan
        else:
            merged[zcol] = (merged[col] - merged[col].mean()) / std

    return merged


def fit_joint_cox(df, interaction=False):
    pc_col = f"AE_{SELECTED_PC}_z"
    use_cols = [
        "OS_time_months", "OS", "cancer_type",
        "age", "sex_bin", "stage_bin", "SMA_zscore", pc_col
    ]
    d = df[use_cols].dropna().copy()
    d = d[d["OS_time_months"] > 0].copy()

    if interaction:
        d["SMA_PC6_interaction"] = d["SMA_zscore"] * d[pc_col]
        exog_cols = ["SMA_zscore", pc_col, "SMA_PC6_interaction", "age", "sex_bin", "stage_bin"]
    else:
        exog_cols = ["SMA_zscore", pc_col, "age", "sex_bin", "stage_bin"]

    model = PHReg(
        endog=d["OS_time_months"],
        exog=d[exog_cols],
        status=d["OS"],
        strata=d["cancer_type"],
        ties="breslow"
    )
    result = model.fit()
    return d, result, exog_cols


def phreg_summary_table(result, exog_cols, cohort_name, model_name):
    rows = []
    for i, var in enumerate(exog_cols):
        beta = float(result.params[i])
        se = float(result.bse[i])
        z = beta / se
        p = 2 * norm.sf(abs(z))
        lo = beta - 1.96 * se
        hi = beta + 1.96 * se

        rows.append({
            "cohort": cohort_name,
            "model": model_name,
            "variable": var,
            "beta": beta,
            "HR": np.exp(beta),
            "CI_low": np.exp(lo),
            "CI_high": np.exp(hi),
            "p_value": p
        })

    return pd.DataFrame(rows)


def lr_test(base_result, full_result, df_diff=1):
    LR = 2 * (full_result.llf - base_result.llf)
    p = chi2.sf(LR, df_diff)
    return LR, p


def km_rmst(time, event, tau):
    sf = SurvfuncRight(time, event)
    surv_times = np.asarray(sf.surv_times, dtype=float)
    surv_prob = np.asarray(sf.surv_prob, dtype=float)

    times = np.concatenate(([0.0], surv_times[surv_times < tau], [tau]))
    surv_times_lt = surv_times[surv_times < tau]
    surv_prob_lt = surv_prob[surv_times < tau]

    area = 0.0
    current_surv = 1.0

    for k in range(len(times) - 1):
        t0, t1 = times[k], times[k + 1]
        area += (t1 - t0) * current_surv
        if k < len(surv_times_lt):
            current_surv = surv_prob_lt[k]

    return area


def make_joint_groups(df, sma_cut, pc_cut):
    d = df.copy()
    pc_col = f"AE_{SELECTED_PC}_z"

    d["SMA_group"] = np.where(d["SMA_zscore"] >= sma_cut, "High SMA", "Low SMA")
    d["PC6_group"] = np.where(d[pc_col] >= pc_cut, "High PC6", "Low PC6")
    d["joint_group"] = d["SMA_group"] + " / " + d["PC6_group"]
    return d


def rmst_table(df, cohort_name, tau):
    order = [
        "High SMA / Low PC6",
        "High SMA / High PC6",
        "Low SMA / Low PC6",
        "Low SMA / High PC6",
    ]

    rows = []
    for g in order:
        dd = df[df["joint_group"] == g].copy()
        if len(dd) == 0:
            continue

        rows.append({
            "cohort": cohort_name,
            "group": g,
            "n": len(dd),
            "events": int(dd["OS"].sum()),
            f"RMST_{int(tau)}m": km_rmst(dd["OS_time_months"], dd["OS"], tau)
        })

    return pd.DataFrame(rows)


def build_risk_surface(result, exog_cols, cohort_name, df_used):
    beta_map = {var: float(result.params[i]) for i, var in enumerate(exog_cols)}
    beta_sma = beta_map["SMA_zscore"]
    beta_pc6 = beta_map[f"AE_{SELECTED_PC}_z"]
    beta_int = beta_map.get("SMA_PC6_interaction", 0.0)

    x_min = np.nanquantile(df_used["SMA_zscore"], 0.01)
    x_max = np.nanquantile(df_used["SMA_zscore"], 0.99)
    y_min = np.nanquantile(df_used[f"AE_{SELECTED_PC}_z"], 0.01)
    y_max = np.nanquantile(df_used[f"AE_{SELECTED_PC}_z"], 0.99)

    x = np.linspace(x_min, x_max, 220)
    y = np.linspace(y_min, y_max, 220)
    X, Y = np.meshgrid(x, y)

    RR = np.exp(beta_sma * X + beta_pc6 * Y + beta_int * X * Y)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cf = ax.contourf(X, Y, RR, levels=14, cmap="RdYlBu_r", alpha=0.95)
    cs = ax.contour(X, Y, RR, levels=8, colors="white", linewidths=0.7, alpha=0.65)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

    ax.scatter(
        df_used["SMA_zscore"],
        df_used[f"AE_{SELECTED_PC}_z"],
        s=16, facecolors="white", edgecolors="#555555",
        linewidths=0.35, alpha=0.35, zorder=3
    )

    ax.axvline(0, color="#333333", lw=1.1, ls="--", alpha=0.8)
    ax.axhline(0, color="#333333", lw=1.1, ls="--", alpha=0.8)
    ax.set_xlabel("SMA z-score")
    ax.set_ylabel(f"{SELECTED_PC} z-score")
    ax.set_title(f"{cohort_name}: risk surface from SMA × {SELECTED_PC} interaction model", fontweight="bold")

    txt = (
        f"Relative hazard = exp(β_SMA*SMA + β_{SELECTED_PC}*{SELECTED_PC} + β_int*SMA*{SELECTED_PC})\n"
        f"β_SMA={beta_sma:.3f}, β_{SELECTED_PC}={beta_pc6:.3f}, β_int={beta_int:.3f}"
    )
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#BBBBBB", alpha=0.92)
    )

    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative hazard")

    out_png = os.path.join(OUTDIR, f"risk_surface_{cohort_name.lower()}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return out_png


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    tcga = pd.read_csv(TCGA_PATH)
    ext = pd.read_csv(EXT_PATH)
    ae_t = pd.read_csv(AE_TCGA)
    ae_e = pd.read_csv(AE_EXT)

    df_t = preprocess_and_merge(tcga, ae_t, "TCGA")
    df_e = preprocess_and_merge(ext, ae_e, "CPTAC")

    # 1) Joint multivariable Cox
    d_t_joint, res_t_joint, cols_t_joint = fit_joint_cox(df_t, interaction=False)
    d_e_joint, res_e_joint, cols_e_joint = fit_joint_cox(df_e, interaction=False)

    tab_t_joint = phreg_summary_table(res_t_joint, cols_t_joint, "TCGA", "joint")
    tab_e_joint = phreg_summary_table(res_e_joint, cols_e_joint, "CPTAC", "joint")

    # 2) Interaction analysis
    d_t_int, res_t_int, cols_t_int = fit_joint_cox(df_t, interaction=True)
    d_e_int, res_e_int, cols_e_int = fit_joint_cox(df_e, interaction=True)

    tab_t_int = phreg_summary_table(res_t_int, cols_t_int, "TCGA", "interaction")
    tab_e_int = phreg_summary_table(res_e_int, cols_e_int, "CPTAC", "interaction")

    lr_t, lr_p_t = lr_test(res_t_joint, res_t_int, df_diff=1)
    lr_e, lr_p_e = lr_test(res_e_joint, res_e_int, df_diff=1)

    interaction_lr = pd.DataFrame([
        {"cohort": "TCGA",  "LR_stat": lr_t, "LR_p_value": lr_p_t},
        {"cohort": "CPTAC", "LR_stat": lr_e, "LR_p_value": lr_p_e},
    ])

    # 3) RMST (use TCGA medians as common cutoffs)
    sma_cut = float(d_t_joint["SMA_zscore"].median())
    pc_cut = float(d_t_joint[f"AE_{SELECTED_PC}_z"].median())

    rmst_t = rmst_table(make_joint_groups(d_t_joint, sma_cut, pc_cut), "TCGA", TAU_MONTHS)
    rmst_e = rmst_table(make_joint_groups(d_e_joint, sma_cut, pc_cut), "CPTAC", TAU_MONTHS)
    rmst_all = pd.concat([rmst_t, rmst_e], ignore_index=True)

    # 4) Risk surface modeling (interaction model)
    risk_png_t = build_risk_surface(res_t_int, cols_t_int, "TCGA", d_t_int)
    risk_png_e = build_risk_surface(res_e_int, cols_e_int, "CPTAC", d_e_int)

    # save tables
    cox_all = pd.concat([tab_t_joint, tab_e_joint, tab_t_int, tab_e_int], ignore_index=True)
    cox_all.to_csv(os.path.join(OUTDIR, "cox_summary_pc6.csv"), index=False)
    interaction_lr.to_csv(os.path.join(OUTDIR, "interaction_lr_test_pc6.csv"), index=False)
    rmst_all.to_csv(os.path.join(OUTDIR, f"rmst_{int(TAU_MONTHS)}m_pc6.csv"), index=False)

    print("Saved:")
    print(os.path.join(OUTDIR, "cox_summary_pc6.csv"))
    print(os.path.join(OUTDIR, "interaction_lr_test_pc6.csv"))
    print(os.path.join(OUTDIR, f"rmst_{int(TAU_MONTHS)}m_pc6.csv"))
    print(risk_png_t)
    print(risk_png_e)