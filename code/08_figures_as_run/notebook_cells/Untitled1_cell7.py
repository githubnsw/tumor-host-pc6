"""
Figure 4. Clinical Impact of PC6 in Combination with Body Composition
=====================================================================
(A) TCGA:   multivariable Cox (SMA vs PC6)
(B) CPTAC:  multivariable Cox (SMA vs PC6)
(C) Risk map (SMA × PC6, TCGA-centered)

수정사항:
  - A, B 변수 라벨: axes 왼쪽 바깥 고정 (CI선과 겹침 없음)
  - HR 텍스트: 점 바로 아래 배치
  - 제목 + cohort info 통합

Input
-----
  TCGA_with_genes.csv
  External_with_genes.csv
  ae_latent_pca10_all.csv
  ae_latent_pca10_external.csv

Output
------
  Figure4_fixed.tiff

Run
---
  python figure4_fixed.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE = r"C:\Users\nsw\Desktop\body_composition"

TCGA_PATH = os.path.join(BASE, "TCGA_with_genes.csv")
EXT_PATH  = os.path.join(BASE, "External_with_genes.csv")
AE_TCGA   = os.path.join(BASE, "ae_latent_pca10_all.csv")
AE_EXT    = os.path.join(BASE, "ae_latent_pca10_external.csv")
OUTPATH   = os.path.join(BASE, "Figure4_fixed.tiff")

SELECTED_PC = "PC6"


# ============================================================
# HELPERS
# ============================================================

def star(p):
    if pd.isna(p): return "na"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def fmt_p(p):
    if pd.isna(p): return "NA"
    if p < 0.001:  return "<0.001"
    return f"{p:.3f}"


def format_stage_binary(x):
    s = str(x).upper()
    if ("III" in s) or ("IV" in s): return 1.0
    if ("II"  in s) or ("I"  in s): return 0.0
    return np.nan


def pretty_var_name(v, selected_pc):
    if v == "SMA_zscore":           return "SMA z-score"
    if v == f"AE_{selected_pc}_z": return selected_pc
    return v


def preprocess_and_merge(clin_df, ae_df, cohort_name):
    df = clin_df.copy()
    ae = ae_df.copy()

    df["age"]        = pd.to_numeric(df["age"],        errors="coerce")
    df["OS"]         = pd.to_numeric(df["OS"],         errors="coerce")
    df["OS_time"]    = pd.to_numeric(df["OS_time"],    errors="coerce") / 30.44
    df["SMA_zscore"] = pd.to_numeric(df["SMA_zscore"], errors="coerce")
    df["sex_bin"]    = (df["sex"].astype(str).str.upper() == "MALE").astype(float)
    df["stage_bin"]  = df["stage"].apply(format_stage_binary)

    if cohort_name == "TCGA":
        df["pid"] = df["patient_id"].astype(str).str[:12]
        ae["pid"] = ae["patient_id"].astype(str).str[:12]
    else:
        df["pid"] = df["patient_id"].astype(str)
        ae["pid"] = ae["patient_id"].astype(str)

    use_cols = ["pid"] + [f"AE_PC{i}" for i in range(1, 11)]
    merged = df.merge(ae[use_cols], on="pid", how="inner")

    for i in range(1, 11):
        col  = f"AE_PC{i}"
        zcol = f"{col}_z"
        std  = merged[col].std()
        if pd.isna(std) or std == 0:
            merged[zcol] = np.nan
        else:
            merged[zcol] = (merged[col] - merged[col].mean()) / std

    return merged


def fit_joint_model(df, selected_pc):
    pc_col   = f"AE_{selected_pc}_z"
    use_cols = [
        "OS_time", "OS", "cancer_type",
        "age", "sex_bin", "stage_bin",
        "SMA_zscore", pc_col
    ]
    tmp = df[use_cols].dropna().copy()
    tmp = tmp[tmp["OS_time"] > 0].copy()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(tmp, duration_col="OS_time", event_col="OS", strata="cancer_type")

    out_rows = []
    for var in ["SMA_zscore", pc_col]:
        s = cph.summary.loc[var]
        out_rows.append({
            "variable": var,
            "label"   : pretty_var_name(var, selected_pc),
            "hr"      : float(s["exp(coef)"]),
            "lo"      : float(s["exp(coef) lower 95%"]),
            "hi"      : float(s["exp(coef) upper 95%"]),
            "p"       : float(s["p"]),
        })
    return cph, tmp, pd.DataFrame(out_rows)


# ════════════════════════════════════════════════════════════════
# Forest plot (A, B 공통)
# ════════════════════════════════════════════════════════════════

def draw_joint_forest(ax, result_df, cohort_label, n, events, xlim, selected_pc):
    plot_df = result_df.copy()
    order   = ["SMA z-score", selected_pc]
    plot_df["order"] = plot_df["label"].map({k: i for i, k in enumerate(order)})
    plot_df = plot_df.sort_values("order").reset_index(drop=True)

    y_positions = np.array([1, 0])
    ax.axvline(1.0, color="#777777", lw=1.1, ls="--", zorder=0)

    for i, row in plot_df.iterrows():
        y      = y_positions[i]
        is_pc  = row["label"] == selected_pc
        color  = "#C0392B" if is_pc else "#2E86C1"
        marker = "s"       if is_pc else "D"
        fc     = "#FDEDEC" if is_pc else "#EBF5FB"
        alpha  = 0.85      if is_pc else 0.75

        ax.axhspan(y - 0.42, y + 0.42, color=fc, alpha=alpha, zorder=0)

        ax.errorbar(
            row["hr"], y,
            xerr=[[row["hr"] - row["lo"]], [row["hi"] - row["hr"]]],
            fmt=marker, color=color,
            markerfacecolor=color, markeredgecolor="#111111",
            markeredgewidth=0.9,
            markersize=8.5 if row["p"] < 0.05 else 6.5,
            elinewidth=1.8 if row["p"] < 0.05 else 1.2,
            capsize=3.5, zorder=3
        )

        # ── 변수 라벨: axes 왼쪽 바깥 고정 (CI선과 겹치지 않음) ──
        ax.text(-0.18, y, row["label"],
                transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=color)

        # ── HR 텍스트: 점 바로 아래 (CI선과 겹치지 않음) ─────────
        hr_text = (f"HR={row['hr']:.2f} ({row['lo']:.2f}–{row['hi']:.2f})\n"
                   f"p={fmt_p(row['p'])} {star(row['p'])}")
        ax.text(row["hr"], y - 0.28, hr_text,
                ha="center", va="top", fontsize=8.0,
                color=color,
                fontweight="bold" if row["p"] < 0.05 else "normal")

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(-0.85, 1.75)
    ax.set_yticks([])
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=11)

    # ── 제목 + cohort info 통합 ────────────────────────────────
    ax.set_title(
        f"{cohort_label}\n"
        f"n={n}, events={events} | adj: age, sex, stage | strat: cancer type",
        fontsize=10.5, fontweight="bold", pad=12, linespacing=1.5
    )

    legend_handles = [
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#2E86C1",
               markeredgecolor="#111111", markersize=7.5, label="SMA z-score"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#C0392B",
               markeredgecolor="#111111", markersize=7.5, label=selected_pc),
        mpatches.Patch(facecolor="#EBF5FB", edgecolor="#2E86C1", label="Body composition"),
        mpatches.Patch(facecolor="#FDEDEC", edgecolor="#C0392B", label="Latent feature"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.95)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)


# ════════════════════════════════════════════════════════════════
# Risk surface (C)
# ════════════════════════════════════════════════════════════════

def build_risk_map(ax, cph_model, tcga_df, selected_pc):
    pc_col   = f"AE_{selected_pc}_z"
    beta_sma = float(cph_model.params_["SMA_zscore"])
    beta_pc  = float(cph_model.params_[pc_col])

    x_min = np.nanquantile(tcga_df["SMA_zscore"], 0.01)
    x_max = np.nanquantile(tcga_df["SMA_zscore"], 0.99)
    y_min = np.nanquantile(tcga_df[pc_col], 0.01)
    y_max = np.nanquantile(tcga_df[pc_col], 0.99)

    x = np.linspace(x_min, x_max, 220)
    y = np.linspace(y_min, y_max, 220)
    X, Y = np.meshgrid(x, y)
    RR   = np.exp(beta_sma * X + beta_pc * Y)

    vmin = np.nanquantile(RR, 0.02)
    vmax = np.nanquantile(RR, 0.98)

    cf = ax.contourf(X, Y, RR, levels=14, cmap="RdYlBu_r",
                     vmin=vmin, vmax=vmax, alpha=0.95)
    cs = ax.contour(X, Y, RR, levels=8, colors="white",
                    linewidths=0.6, alpha=0.55)
    ax.clabel(cs, inline=True, fontsize=6.5, fmt="%.2f")

    ax.scatter(tcga_df["SMA_zscore"], tcga_df[pc_col],
               s=10, facecolors="white", edgecolors="#999999",
               linewidths=0.25, alpha=0.45, zorder=3)

    ax.axvline(0, color="#444444", lw=0.9, ls="--", alpha=0.8)
    ax.axhline(0, color="#444444", lw=0.9, ls="--", alpha=0.8)

    ax.set_xlabel("SMA z-score",          fontsize=10.8)
    ax.set_ylabel(f"{selected_pc} z-score", fontsize=10.8)
    ax.set_title("(C) TCGA-centered risk map\nCombined effect of SMA and PC6",
                 fontsize=11.0, fontweight="bold", pad=16)

    ax.text(0.03, 0.99,
            "Relative hazard from TCGA joint Cox model\n"
            r"$RR = \exp(\beta_{SMA}\cdot SMA_z + \beta_{PC6}\cdot PC6_z)$",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#BBBBBB", alpha=0.92))

    corners = [
        {"name": "Low SMA\nHigh PC6",  "x": x_min, "y": y_max},
        {"name": "High SMA\nHigh PC6", "x": x_max, "y": y_max},
        {"name": "Low SMA\nLow PC6",   "x": x_min, "y": y_min},
        {"name": "High SMA\nLow PC6",  "x": x_max, "y": y_min},
    ]
    for c in corners:
        c["rr"] = np.exp(beta_sma * c["x"] + beta_pc * c["y"])

    highest = max(corners, key=lambda z: z["rr"])
    lowest  = min(corners, key=lambda z: z["rr"])
    x_pad   = 0.06 * (x_max - x_min)
    y_pad   = 0.06 * (y_max - y_min)

    ax.text(highest["x"] + x_pad, highest["y"] - y_pad,
            f"Highest risk\n{highest['name']}",
            ha="center", va="center", fontsize=8.3, fontweight="bold",
            color="#7B241C",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#FDEDEC",
                      edgecolor="#C0392B", alpha=0.95))

    ax.text(lowest["x"] - x_pad, lowest["y"] + y_pad,
            f"Lowest risk\n{lowest['name']}",
            ha="center", va="center", fontsize=8.3, fontweight="bold",
            color="#1B4F72",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#EBF5FB",
                      edgecolor="#2E86C1", alpha=0.95))

    cbar = plt.colorbar(cf, ax=ax, fraction=0.050, pad=0.03)
    cbar.set_label("Relative hazard", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
tcga = pd.read_csv(TCGA_PATH)
ext  = pd.read_csv(EXT_PATH)
ae_t = pd.read_csv(AE_TCGA)
ae_e = pd.read_csv(AE_EXT)

df_tcga = preprocess_and_merge(tcga, ae_t, cohort_name="TCGA")
df_ext  = preprocess_and_merge(ext,  ae_e, cohort_name="CPTAC")
print(f"  TCGA merged n = {len(df_tcga)}")
print(f"  CPTAC merged n = {len(df_ext)}")

# ============================================================
# FIT JOINT MODELS
# ============================================================
print("Fitting joint Cox models...")
tcga_model, tcga_used, tcga_res = fit_joint_model(df_tcga, SELECTED_PC)
ext_model,  ext_used,  ext_res  = fit_joint_model(df_ext,  SELECTED_PC)

print("\n[TCGA joint model]")
for _, row in tcga_res.iterrows():
    print(f"  {row['label']}: HR={row['hr']:.3f} ({row['lo']:.3f}-{row['hi']:.3f})  p={row['p']:.4f}")

print("\n[CPTAC joint model]")
for _, row in ext_res.iterrows():
    print(f"  {row['label']}: HR={row['hr']:.3f} ({row['lo']:.3f}-{row['hi']:.3f})  p={row['p']:.4f}")

# ============================================================
# COMMON X LIMIT
# ============================================================
all_lo = pd.concat([tcga_res["lo"], ext_res["lo"]]).replace([np.inf,-np.inf], np.nan).dropna()
all_hi = pd.concat([tcga_res["hi"], ext_res["hi"]]).replace([np.inf,-np.inf], np.nan).dropna()
xmin = max(0.35, float(all_lo.min()) * 0.88)
xmax = max(2.15, float(all_hi.max()) * 1.22)
xlim = (xmin, xmax)

# ============================================================
# PLOT
# ============================================================
fig = plt.figure(figsize=(24, 9))
fig.patch.set_facecolor("white")
gs = GridSpec(1, 3, figure=fig,
              width_ratios=[1.05, 1.05, 1.12],
              wspace=0.55,
              left=0.10, right=0.97,
              top=0.84, bottom=0.12)

# Panel A
ax1 = fig.add_subplot(gs[0, 0])
draw_joint_forest(
    ax=ax1,
    result_df=tcga_res,
    cohort_label="(A) TCGA: multivariable Cox model",
    n=len(tcga_used),
    events=int(tcga_used["OS"].sum()),
    xlim=xlim,
    selected_pc=SELECTED_PC
)

# Panel B
ax2 = fig.add_subplot(gs[0, 1])
draw_joint_forest(
    ax=ax2,
    result_df=ext_res,
    cohort_label="(B) CPTAC-3: multivariable Cox model",
    n=len(ext_used),
    events=int(ext_used["OS"].sum()),
    xlim=xlim,
    selected_pc=SELECTED_PC
)

# Panel C
ax3 = fig.add_subplot(gs[0, 2])
build_risk_map(
    ax=ax3,
    cph_model=tcga_model,
    tcga_df=tcga_used,
    selected_pc=SELECTED_PC
)

# ============================================================
# TITLE
# ============================================================
fig.suptitle(
    "Figure 4.  Clinical Impact of PC6 in Combination with Body Composition",
    fontsize=15, fontweight="bold", y=0.97
)

plt.savefig(OUTPATH, dpi=300, bbox_inches="tight", facecolor="white", format="tiff")
plt.close()

print(f"\nSaved: {OUTPATH}")