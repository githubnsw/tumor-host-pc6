"""
Figure 3. Selection of PC6 as the Key Prognostic Latent Feature
================================================================
Panel A: Forest plot  — HR ± 95%CI, TCGA + CPTAC-3
Panel B: -log10(p) scatter — bilateral significance
(Panel C 제거)

수정사항: 전체 글씨 크기 키움

입력:  TCGA_with_genes.csv, External_with_genes.csv
       ae_latent_pca10_all.csv, ae_latent_pca10_external.csv
출력:  Figure3.tiff  (dpi=300)
실행:  python figure3.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIG
# ============================================================
BASE      = r"C:\Users\nsw\Desktop\body_composition"
TCGA_PATH = os.path.join(BASE, "TCGA_with_genes.csv")
EXT_PATH  = os.path.join(BASE, "External_with_genes.csv")
AE_TCGA   = os.path.join(BASE, "ae_latent_pca10_all.csv")
AE_EXT    = os.path.join(BASE, "ae_latent_pca10_external.csv")
OUTPATH   = os.path.join(BASE, "Figure3.tiff")


# ============================================================
# Helpers
# ============================================================
def star(p):
    if pd.isna(p): return 'na'
    if p < 0.001:  return '***'
    if p < 0.01:   return '**'
    if p < 0.05:   return '*'
    return 'ns'

def safe_neglog10(p):
    if pd.isna(p): return 0.0
    return -np.log10(max(float(p), 1e-300))

def same_direction(hr1, hr2):
    if pd.isna(hr1) or pd.isna(hr2): return False
    return ((hr1 > 1) and (hr2 > 1)) or ((hr1 < 1) and (hr2 < 1))

def fmt_hr(hr, lo, hi, p):
    if any(pd.isna(x) for x in [hr, lo, hi, p]): return "NA"
    return f"HR={hr:.3f} ({lo:.3f}–{hi:.3f})  {star(p)}"

def choose_selected_pc(screen_df):
    df = screen_df.copy()
    df['t_sig']       = df['t_p'] < 0.05
    df['e_sig']       = df['e_p'] < 0.05
    df['same_dir']    = [same_direction(t_hr, e_hr)
                         for t_hr, e_hr in zip(df['t_hr'], df['e_hr'])]
    df['bilateral']   = df['t_sig'] & df['e_sig'] & df['same_dir']
    df['score']       = df['t_p'].apply(safe_neglog10) + df['e_p'].apply(safe_neglog10)
    df['effect_score']= np.abs(df['t_hr'] - 1) + np.abs(df['e_hr'] - 1)
    bilateral_df = df[df['bilateral']].copy()
    if len(bilateral_df) == 0:
        return None, df
    bilateral_df = bilateral_df.sort_values(
        by=['score', 'effect_score'], ascending=[False, False]
    ).reset_index(drop=True)
    return bilateral_df.loc[0, 'PC'], df


# ============================================================
# Load & prep
# ============================================================
print("Loading data...")
t    = pd.read_csv(TCGA_PATH)
e    = pd.read_csv(EXT_PATH)
ae_t = pd.read_csv(AE_TCGA)
ae_e = pd.read_csv(AE_EXT)

for df in [t, e]:
    df['age']       = pd.to_numeric(df['age'], errors='coerce')
    df['OS_time']   = pd.to_numeric(df['OS_time'], errors='coerce') / 30.44
    df['sex_bin']   = (df['sex'].astype(str).str.upper() == 'MALE').astype(float)
    df['stage_bin'] = df['stage'].apply(
        lambda s: 1.0 if any(x in str(s) for x in ['III', 'IV'])
        else (0.0 if any(x in str(s) for x in ['I', 'II']) else np.nan))

t['pid']    = t['patient_id'].astype(str).str[:12]
ae_t['pid'] = ae_t['patient_id'].astype(str).str[:12]
e['pid']    = e['patient_id'].astype(str)
ae_e['pid'] = ae_e['patient_id'].astype(str)

for df_clin, ae_df, name in [(t, ae_t, 't'), (e, ae_e, 'e')]:
    merged = df_clin.merge(
        ae_df[['pid'] + [f'AE_PC{i}' for i in range(1, 11)]],
        on='pid', how='inner')
    for i in range(1, 11):
        col = f'AE_PC{i}'
        std = merged[col].std()
        merged[f'{col}_z'] = (
            (merged[col] - merged[col].mean()) / std
            if (not pd.isna(std) and std > 0) else np.nan)
    if name == 't': df_t = merged
    else:           df_e = merged

print(f"  TCGA: {len(df_t)}  |  CPTAC-3: {len(df_e)}")


# ============================================================
# Multivariable Cox: each PC
# ============================================================
print("Computing Cox...")
rows = []

for i in range(1, 11):
    pc_col = f'AE_PC{i}_z'
    row = {'PC': f'PC{i}'}
    for tag, df_ in [('t', df_t), ('e', df_e)]:
        d = df_[['OS_time', 'OS', 'cancer_type', 'age', 'sex_bin',
                  'stage_bin', 'SMA_zscore', pc_col]].dropna()
        d = d[d['OS_time'] > 0]
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(d, duration_col='OS_time', event_col='OS', strata='cancer_type')
            r = cph.summary.loc[pc_col]
            row[f'{tag}_hr'] = r['exp(coef)']
            row[f'{tag}_lo'] = r['exp(coef) lower 95%']
            row[f'{tag}_hi'] = r['exp(coef) upper 95%']
            row[f'{tag}_p']  = r['p']
        except Exception:
            row[f'{tag}_hr'] = np.nan; row[f'{tag}_lo'] = np.nan
            row[f'{tag}_hi'] = np.nan; row[f'{tag}_p']  = 1.0
    print(f"  PC{i}: TCGA HR={row['t_hr']:.3f} p={row['t_p']:.4f} | "
          f"CPTAC-3 HR={row['e_hr']:.3f} p={row['e_p']:.4f}")
    rows.append(row)

SCREEN = pd.DataFrame(rows)
selected_pc, SCREEN = choose_selected_pc(SCREEN)

selected_idx = None
sel_row      = None
if selected_pc:
    selected_idx   = SCREEN.index[SCREEN['PC'] == selected_pc][0]
    sel_row        = SCREEN[SCREEN['PC'] == selected_pc].iloc[0]
    direction_text = "same direction" if sel_row['same_dir'] else "discordant direction"
else:
    direction_text = "no replicated signal"

SIG = -np.log10(0.05)
print(f"\nSelected PC: {selected_pc}")


# ============================================================
# Figure  (1 row × 2 panels — C 제거)
# ============================================================
fig = plt.figure(figsize=(22, 10))
fig.patch.set_facecolor('white')

gs = GridSpec(1, 2, figure=fig,
              wspace=0.48, left=0.07, right=0.97,
              top=0.86, bottom=0.10,
              width_ratios=[1.2, 1.0])


# ──────────────────────────────────────────────────────────────
# Panel A: Forest plot
# ──────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
n = len(SCREEN); ys = np.arange(n)[::-1]; OFF = 0.20
ax_a.axvline(1, color='#888', lw=1.2, ls='--', zorder=0)

for i, row in SCREEN.iterrows():
    y      = ys[i]
    is_sel = bool(selected_pc and row['PC'] == selected_pc)
    t_s    = row['t_sig']; e_s = row['e_sig']

    if is_sel:
        ax_a.axhspan(y-0.46, y+0.46, color='#FADBD8', alpha=0.45, zorder=1)
    elif i % 2 == 0:
        ax_a.axhspan(y-0.46, y+0.46, color='#F8F9FA', alpha=0.4,  zorder=0)

    # TCGA
    tc = '#2980B9' if t_s else '#AED6F1'
    ax_a.errorbar(row['t_hr'], y+OFF,
                  xerr=[[row['t_hr']-row['t_lo']], [row['t_hi']-row['t_hr']]],
                  fmt='D' if t_s else 'o', color=tc,
                  markersize=12 if (t_s and is_sel) else (10 if t_s else 6),
                  elinewidth=2.2 if t_s else 1.0, capsize=4, zorder=4,
                  markeredgecolor='#111' if t_s else tc,
                  markeredgewidth=0.9 if t_s else 0)

    # CPTAC-3
    ec = '#E74C3C' if e_s else '#F1948A'
    ax_a.errorbar(row['e_hr'], y-OFF,
                  xerr=[[row['e_hr']-row['e_lo']], [row['e_hi']-row['e_hr']]],
                  fmt='s' if e_s else 'o', color=ec,
                  markersize=12 if (e_s and is_sel) else (10 if e_s else 6),
                  elinewidth=2.2 if e_s else 1.0, capsize=4, zorder=4,
                  markeredgecolor='#111' if e_s else ec,
                  markeredgewidth=0.9 if e_s else 0)

    ax_a.text(-0.02, y, row['PC'], va='center', ha='right', fontsize=13,
              fontweight='bold' if is_sel else 'normal',
              color='#C0392B' if is_sel else '#2C3E50',
              transform=ax_a.get_yaxis_transform())
    ax_a.text(2.90, y+OFF,
              fmt_hr(row['t_hr'], row['t_lo'], row['t_hi'], row['t_p']),
              va='center', ha='left', fontsize=10.5,
              color='#2980B9' if t_s else '#CCC',
              fontweight='bold' if t_s else 'normal')
    ax_a.text(2.90, y-OFF,
              fmt_hr(row['e_hr'], row['e_lo'], row['e_hi'], row['e_p']),
              va='center', ha='left', fontsize=10.5,
              color='#E74C3C' if e_s else '#CCC',
              fontweight='bold' if e_s else 'normal')
    if is_sel:
        for yo, co in [(OFF, '#2980B9'), (-OFF, '#E74C3C')]:
            ax_a.text(0.03, y+yo, '★', va='center', ha='left',
                      fontsize=13, color=co, zorder=5,
                      transform=ax_a.get_yaxis_transform())

ax_a.set_xlim(0.45, 2.85); ax_a.set_ylim(-0.8, n-0.15); ax_a.set_yticks([])
ax_a.set_xlabel('Hazard Ratio  (95% CI)', fontsize=14)
ax_a.tick_params(axis='x', labelsize=12)
ax_a.set_title('PC1–PC10 Multivariable Cox Forest Plot\n'
               'adj: SMA, age, sex, stage  |  strat: cancer type  |  OS',
               fontsize=14, fontweight='bold')
ax_a.text(2.88, n-0.42, 'HR (95% CI)   sig',
          va='center', ha='left', fontsize=11, fontweight='bold', color='#555')

leg_a = [
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#2980B9',
           markeredgecolor='#111', markersize=11, label='TCGA  p<0.05'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#E74C3C',
           markeredgecolor='#111', markersize=11, label='CPTAC-3  p<0.05'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#BDC3C7',
           markersize=8, label='ns'),
]
if selected_pc:
    leg_a.append(mpatches.Patch(facecolor='#FADBD8', alpha=0.6,
                                edgecolor='#C0392B', label=f'{selected_pc}  (selected)'))
ax_a.legend(handles=leg_a, fontsize=11, framealpha=0.9, loc='lower right')
for sp in ['top', 'right', 'left']: ax_a.spines[sp].set_visible(False)
ax_a.text(-0.06, 1.05, 'A', transform=ax_a.transAxes, fontsize=17, fontweight='bold')


# ──────────────────────────────────────────────────────────────
# Panel B: Bilateral significance scatter
# ──────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
t_lp = [safe_neglog10(p) for p in SCREEN['t_p']]
e_lp = [safe_neglog10(p) for p in SCREEN['e_p']]
eff  = np.abs(SCREEN['t_hr'].values-1) + np.abs(SCREEN['e_hr'].values-1)
eff  = np.nan_to_num(eff, nan=0.0)
sz   = eff * 700 + 60

xmax = max(max(t_lp)*1.5, SIG+0.8, 2.0)
ymax = max(max(e_lp)*1.9, SIG+0.8, 2.0)

ax_b.fill_between([SIG, xmax], SIG, ymax, color='#FADBD8', alpha=0.25, zorder=0)
ax_b.axvline(SIG, color='#2980B9', lw=1.2, ls='--', alpha=0.65)
ax_b.axhline(SIG, color='#E74C3C', lw=1.2, ls='--', alpha=0.65)
ax_b.text(SIG+0.04, 0.05, 'TCGA p=0.05',
          color='#2980B9', fontsize=11, rotation=90, va='bottom')
ax_b.text(0.05, SIG+0.04, 'CPTAC-3 p=0.05',
          color='#E74C3C', fontsize=11, va='bottom')

for i, row in SCREEN.iterrows():
    is_sel = bool(selected_pc and row['PC'] == selected_pc)
    t_s = row['t_sig']; e_s = row['e_sig']; bil = row['bilateral']
    if is_sel:            col = '#C0392B'
    elif bil:             col = '#27AE60'
    elif t_s and not e_s: col = '#2980B9'
    elif e_s and not t_s: col = '#9B59B6'
    else:                 col = '#BDC3C7'

    ax_b.scatter(t_lp[i], e_lp[i], s=sz[i], c=col,
                 zorder=4 if is_sel else 3,
                 edgecolors='#111' if is_sel else col,
                 linewidths=1.8 if is_sel else 0.4, alpha=0.90)
    ax_b.annotate(row['PC'],
                  xy=(t_lp[i], e_lp[i]),
                  xytext=(t_lp[i]+0.06, e_lp[i]+0.06),
                  fontsize=12.5 if is_sel else 11.5,
                  fontweight='bold' if is_sel else 'normal',
                  color='#C0392B' if is_sel else '#444')

if selected_pc and sel_row is not None:
    ax_b.annotate(
        (f'★  {selected_pc}\n'
         f'TCGA p={sel_row["t_p"]:.3f} {star(sel_row["t_p"])}\n'
         f'CPTAC-3 p={sel_row["e_p"]:.3f} {star(sel_row["e_p"])}\n'
         f'→ Bilateral + {direction_text}'),
        xy=(t_lp[selected_idx], e_lp[selected_idx]),
        xytext=(t_lp[selected_idx]+0.40, e_lp[selected_idx]+0.55),
        fontsize=11.5, color='#C0392B', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.30', facecolor='#FADBD8',
                  edgecolor='#C0392B', alpha=0.92),
        arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.8))

ax_b.set_xlabel('−log₁₀(p)  TCGA', fontsize=14)
ax_b.set_ylabel('−log₁₀(p)  CPTAC-3', fontsize=14)
ax_b.tick_params(axis='both', labelsize=12)
ax_b.set_xlim(0, xmax); ax_b.set_ylim(0, ymax)
ax_b.set_title('Bilateral Significance\nTCGA vs CPTAC-3', fontsize=14, fontweight='bold')

for sp in ['top', 'right']: ax_b.spines[sp].set_visible(False)
ax_b.text(-0.12, 1.05, 'B', transform=ax_b.transAxes, fontsize=17, fontweight='bold')


# ============================================================
# Figure title
# ============================================================
if selected_pc:
    fig_title = (
        f'Selection of the Key Prognostic Latent Feature \n'
    )
else:
    fig_title = (
        'Selection of PC6 as the Key Prognostic Latent Feature\n'
        'No component satisfied bilateral prognostic replication with concordant effect direction.'
    )

fig.suptitle(fig_title, fontsize=16, fontweight='bold', y=0.97)

plt.savefig(OUTPATH, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
plt.close()

print(f"Saved: {OUTPATH}")