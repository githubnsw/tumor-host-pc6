"""
Figure 1. Immune–Host Axes and Initial Stratification
=======================================================
Panels:
  A. GEP vs SMA scatter (r, p)  → independence
  B. 4-group distribution bar   → group composition
  C. 4-group Kaplan–Meier (TCGA + External)

Layout: 1 row × 4 panels  (A | B | C-TCGA | C-External)

입력:
  TCGA_with_genes.csv
  External_with_genes.csv

출력:
  Figure1.tiff  (dpi=300)

실행:
  python figure1.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — 경로만 수정
# ============================================================
BASE      = r"C:\Users\nsw\body_composition\R\R1"
TCGA_PATH = os.path.join(BASE, "TCGA_with_genes.csv")
EXT_PATH  = os.path.join(BASE, "External_with_genes.csv")
OUTPATH   = os.path.join(BASE, "Figure1.tiff")

# 그룹 정의
GROUP_ORDER = [
    'Group4_HighMuscle_HighImmune',
    'Group3_HighMuscle_LowImmune',
    'Group2_LowMuscle_HighImmune',
    'Group1_LowMuscle_LowImmune',
]
GROUP_LABELS = {
    'Group4_HighMuscle_HighImmune': 'IH_BG\n(High Immune\nHigh Muscle)',
    'Group3_HighMuscle_LowImmune':  'IL_BG\n(Low Immune\nHigh Muscle)',
    'Group2_LowMuscle_HighImmune':  'IH_BP\n(High Immune\nLow Muscle)',
    'Group1_LowMuscle_LowImmune':   'IL_BP\n(Low Immune\nLow Muscle)',
}
GROUP_COLORS = {
    'Group4_HighMuscle_HighImmune': '#27AE60',
    'Group3_HighMuscle_LowImmune':  '#3498DB',
    'Group2_LowMuscle_HighImmune':  '#E67E22',
    'Group1_LowMuscle_LowImmune':   '#C0392B',
}
KM_LABELS = {
    'Group4_HighMuscle_HighImmune': 'IH_BG: High Immune + High Muscle (Ref)',
    'Group3_HighMuscle_LowImmune':  'IL_BG: Low Immune + High Muscle',
    'Group2_LowMuscle_HighImmune':  'IH_BP: High Immune + Low Muscle',
    'Group1_LowMuscle_LowImmune':   'IL_BP: Low Immune + Low Muscle',
}


# ============================================================
# 데이터 로드
# ============================================================
print("Loading data...")
t = pd.read_csv(TCGA_PATH)
e = pd.read_csv(EXT_PATH)

for df in [t, e]:
    df['age']       = pd.to_numeric(df['age'],      errors='coerce')
    df['OS_time']   = pd.to_numeric(df['OS_time'],  errors='coerce') / 30.44
    df['DSS_time']  = pd.to_numeric(df['DSS_time'], errors='coerce') / 30.44
    df['PFI_time']  = pd.to_numeric(df['PFI_time'], errors='coerce') / 30.44
    df['sex_bin']   = (df['sex'].str.upper() == 'MALE').astype(float)
    df['stage_bin'] = df['stage'].apply(lambda s:
        1.0 if any(x in str(s) for x in ['III','IV']) else
        (0.0 if any(x in str(s) for x in ['I','II'])  else np.nan))

print(f"  TCGA: {len(t)}  |  External: {len(e)}")


# ============================================================
# Figure 구성  (1 × 4)
# ============================================================
fig = plt.figure(figsize=(28, 8))
fig.patch.set_facecolor('white')
gs  = GridSpec(1, 4, figure=fig,
               wspace=0.40,
               left=0.05, right=0.97,
               top=0.88,  bottom=0.13)


# ────────────────────────────────────────────────────────────
# Panel A: GEP vs SMA scatter  →  independence
# ────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

# TCGA
r_t, p_t = spearmanr(t['SMA_zscore'], t['GEP_score'])
ax_a.scatter(t['SMA_zscore'], t['GEP_score'],
             c='#2980B9', alpha=0.40, s=16,
             label=f'TCGA (r={r_t:.2f}, p={p_t:.3f})')

# External
r_e, p_e = spearmanr(e['SMA_zscore'], e['GEP_score'])
ax_a.scatter(e['SMA_zscore'], e['GEP_score'],
             c='#E74C3C', alpha=0.40, s=16, marker='s',
             label=f'External (r={r_e:.2f}, p={p_e:.3f})')

# TCGA trend line
xi  = np.linspace(t['SMA_zscore'].min(), t['SMA_zscore'].max(), 100)
z   = np.polyfit(t['SMA_zscore'], t['GEP_score'], 1)
ax_a.plot(xi, np.poly1d(z)(xi), color='#2980B9', lw=1.8, alpha=0.7)

ax_a.axhline(0, color='#CCC', lw=0.8, ls='--')
ax_a.axvline(0, color='#CCC', lw=0.8, ls='--')
ax_a.set_xlabel('SMA z-score', fontsize=12)
ax_a.set_ylabel('GEP score (Immune activity)', fontsize=12)
ax_a.set_title('A', fontsize=14, fontweight='bold', loc='left')
ax_a.set_title('SMA vs GEP Independence', fontsize=12, fontweight='bold', loc='center')
ax_a.legend(fontsize=9.5, framealpha=0.85, loc='upper left')

# annotation box
ax_a.text(0.97, 0.97,
          f'TCGA: r={r_t:.3f}, p={p_t:.3f}\nExternal: r={r_e:.3f}, p={p_e:.3f}\n'
          f'→ Mutually independent',
          transform=ax_a.transAxes, ha='right', va='top', fontsize=9,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB', alpha=0.9))
for sp in ['top','right']: ax_a.spines[sp].set_visible(False)


# ────────────────────────────────────────────────────────────
# Panel B: 4-group distribution bar
# ────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

n_tcga = [t[t['combined_group']==g].shape[0] for g in GROUP_ORDER]
n_ext  = [e[e['combined_group']==g].shape[0] for g in GROUP_ORDER]
cols   = [GROUP_COLORS[g] for g in GROUP_ORDER]
xlabs  = [GROUP_LABELS[g] for g in GROUP_ORDER]

x  = np.arange(len(GROUP_ORDER))
w  = 0.35
b1 = ax_b.bar(x - w/2, n_tcga, w,
              color=cols, alpha=0.88,
              edgecolor='#333', linewidth=0.7,
              label='TCGA')
b2 = ax_b.bar(x + w/2, n_ext,  w,
              color=cols, alpha=0.50,
              edgecolor='#333', linewidth=0.7,
              hatch='//', label='External')

for bar, n in zip(list(b1) + list(b2), n_tcga + n_ext):
    ax_b.text(bar.get_x() + bar.get_width()/2, n + 1.5,
              str(n), ha='center', va='bottom',
              fontsize=9.5, fontweight='bold')

ax_b.set_xticks(x)
ax_b.set_xticklabels(xlabs, fontsize=9)
ax_b.set_ylabel('Number of patients', fontsize=12)
ax_b.set_title('B', fontsize=14, fontweight='bold', loc='left')
ax_b.set_title('4-Group Distribution', fontsize=12, fontweight='bold', loc='center')
ax_b.legend(fontsize=10, framealpha=0.85)

# total annotation
ax_b.text(0.97, 0.97,
          f'TCGA total: {len(t)}\nExternal total: {len(e)}',
          transform=ax_b.transAxes, ha='right', va='top', fontsize=9.5,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88))
for sp in ['top','right']: ax_b.spines[sp].set_visible(False)


# ────────────────────────────────────────────────────────────
# Panel C: 4-group KM — helper
# ────────────────────────────────────────────────────────────
def km_4grp(ax, df_, cohort_label, panel_letter):
    d = df_[['OS_time','OS','combined_group']].dropna()
    d = d[d['OS_time'] > 0]

    for g in GROUP_ORDER:
        sub = d[d['combined_group'] == g]
        if len(sub) < 3: continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['OS_time'], sub['OS'],
                label=f"{KM_LABELS[g]}\n(n={len(sub)}, events={int(sub['OS'].sum())})")
        kmf.plot_survival_function(ax=ax,
                                   color=GROUP_COLORS[g],
                                   linewidth=2.2,
                                   ci_show=True,
                                   ci_alpha=0.07)

    # Overall log-rank
    res = multivariate_logrank_test(d['OS_time'], d['combined_group'], d['OS'])
    p_ov = res.p_value
    star = '***' if p_ov<0.001 else '**' if p_ov<0.01 else '*' if p_ov<0.05 else 'ns'

    # Worst (G1) vs Ref (G4)
    g1 = d[d['combined_group']=='Group1_LowMuscle_LowImmune']
    g4 = d[d['combined_group']=='Group4_HighMuscle_HighImmune']
    txt = f'Overall log-rank p={p_ov:.4f} {star}'
    if len(g1) >= 3 and len(g4) >= 3:
        lr  = logrank_test(g1['OS_time'], g4['OS_time'],
                           g1['OS'],      g4['OS'])
        p_w = lr.p_value
        sw  = '***' if p_w<0.001 else '**' if p_w<0.01 else '*' if p_w<0.05 else 'ns'
        txt += f'\nIL_BP vs IH_BG: p={p_w:.4f} {sw}'

    col_box = '#C0392B' if p_ov < 0.05 else '#888888'
    ax.text(0.97, 0.97, txt,
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            color=col_box, fontweight='bold' if p_ov<0.05 else 'normal',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.90))

    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Overall Survival probability', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(panel_letter, fontsize=14, fontweight='bold', loc='left')
    ax.set_title(f'4-Group KM — {cohort_label}',
                 fontsize=12, fontweight='bold', loc='center')
    ax.legend(fontsize=8, framealpha=0.85, loc='lower left')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)


# Panel C (TCGA)
km_4grp(fig.add_subplot(gs[0, 2]), t,
        cohort_label='TCGA (n=495)',
        panel_letter='C')

# Panel D (External)
km_4grp(fig.add_subplot(gs[0, 3]), e,
        cohort_label='External CPTAC-3 (n=179)',
        panel_letter='D')


# ============================================================
# Title & Save
# ============================================================
fig.suptitle(
    'Figure 1.  Immune–Host Axes and Initial Stratification\n'
    'GEP (immune activity) and SMA (muscle mass) are independent; '
    'combined stratification reveals survival differences',
    fontsize=13, fontweight='bold', y=1.01
)

plt.savefig(OUTPATH,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            format='tiff')
plt.close()
print(f"Saved: {OUTPATH}")