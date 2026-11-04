"""
Figure 1. Independence of Tumor Immune Activity and Host Body Composition
=========================================================================
Panel A: TCGA  — GEP vs SMA scatter
Panel B: CPTAC-3 — GEP vs SMA scatter
Panel C: Per-cancer-type Spearman r barplot
Panel D: Adjusted Cox forest (GEP, SMA, SMA×GEP)

수정사항:
  - Figure 제목: Figure 1. Independence of Tumor Immune Activity and Host Body Composition
  - A, B 박스: r = 0.06, p = 0.18 (ns) → No correlation
  - A, B 축: GEP score (Tumor immune activity) / SMA z-score (Host body composition)
  - C 하단: No significant correlation across all cancer types

입력:
  TCGA_with_genes.csv
  External_with_genes.csv

출력:
  Figure1.tiff  (dpi=300)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    파일 경로 설정                              ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_DIR   = r"C:\Users\nsw\Desktop\body_composition"
TCGA_FILE  = os.path.join(BASE_DIR, "TCGA_with_genes.csv")
CPTAC_FILE = os.path.join(BASE_DIR, "External_with_genes.csv")
OUT_TIFF   = os.path.join(BASE_DIR, "Figure1.tiff")

# ╔══════════════════════════════════════════════════════════════╗
# ║                    데이터 로드                                 ║
# ╚══════════════════════════════════════════════════════════════╝

tcga  = pd.read_csv(TCGA_FILE)
cptac = pd.read_csv(CPTAC_FILE)

print(f"TCGA  : {len(tcga)} patients")
print(f"CPTAC : {len(cptac)} patients")

# ╔══════════════════════════════════════════════════════════════╗
# ║                    전처리                                      ║
# ╚══════════════════════════════════════════════════════════════╝

def clean_stage(s):
    if pd.isna(s): return np.nan
    s = str(s).upper()
    if 'IV'  in s: return 4
    if 'III' in s: return 3
    if 'II'  in s: return 2
    if 'I'   in s: return 1
    return np.nan

for df in [tcga, cptac]:
    df['stage_num'] = df['stage'].apply(clean_stage)
    df['sex_num']   = (df['sex'].str.upper() == 'MALE').astype(float)

def pstar(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ╔══════════════════════════════════════════════════════════════╗
# ║                    색상 팔레트                                 ║
# ╚══════════════════════════════════════════════════════════════╝

TCGA_COLORS = {
    'KIRC':'#2166AC', 'BLCA':'#4DAF4A', 'LIHC':'#FF7F00',
    'UCEC':'#E41A1C', 'STAD':'#984EA3', 'KIRP':'#A65628',
    'COAD':'#F781BF', 'READ':'#999999'
}
CPTAC_COLORS = {
    'Uterus, NOS':'#E41A1C', 'Pancreas':'#FF7F00', 'Kidney':'#2166AC'
}
coh_colors = {'TCGA':'#4C72B0', 'CPTAC-3':'#DD8452'}

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Adjusted Cox 계산                           ║
# ╚══════════════════════════════════════════════════════════════╝

cox_data = []
for name, df in [('TCGA', tcga), ('CPTAC-3', cptac)]:
    df2 = df[['OS_time','OS','GEP_score','SMA_zscore',
               'age','sex_num','stage_num']].dropna()
    df2 = df2[df2['OS_time'] > 0].copy()
    df2['SMA_x_GEP'] = df2['GEP_score'] * df2['SMA_zscore']
    covs = ['GEP_score','SMA_zscore','SMA_x_GEP','age','stage_num']
    if df2['sex_num'].std() > 0.05:
        covs.append('sex_num')
    fit_df = df2[['OS_time','OS'] + covs].copy()
    cph = CoxPHFitter()
    cph.fit(fit_df, duration_col='OS_time', event_col='OS', robust=True)
    s = cph.summary
    for var, label in [('GEP_score', 'GEP Score'),
                        ('SMA_zscore', 'SMA z-score'),
                        ('SMA_x_GEP', 'SMA × GEP')]:
        row = s.loc[var]
        cox_data.append({
            'cohort'  : name,
            'variable': label,
            'hr'      : row['exp(coef)'],
            'lo'      : row['exp(coef) lower 95%'],
            'hi'      : row['exp(coef) upper 95%'],
            'p'       : row['p'],
            'star'    : pstar(row['p'])
        })

df_cox = pd.DataFrame(cox_data)

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Spearman r 계산                             ║
# ╚══════════════════════════════════════════════════════════════╝

r_t, p_t = stats.spearmanr(tcga['GEP_score'], tcga['SMA_zscore'])
r_c, p_c = stats.spearmanr(cptac['GEP_score'], cptac['SMA_zscore'])

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 레이아웃                             ║
# ╚══════════════════════════════════════════════════════════════╝

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('white')
gs = GridSpec(2, 2, figure=fig,
              hspace=0.42, wspace=0.32,
              left=0.08, right=0.97, top=0.91, bottom=0.08)

LS, TS, TIS, PS = 11, 9.5, 12, 14

# ════════════════════════════════════════════════════════════════
# (A) TCGA: GEP vs SMA scatter
# ════════════════════════════════════════════════════════════════

ax_a = fig.add_subplot(gs[0, 0])

for ct, grp in tcga.groupby('cancer_type'):
    ax_a.scatter(grp['GEP_score'], grp['SMA_zscore'],
                 c=TCGA_COLORS.get(ct, '#888'), alpha=0.6, s=22,
                 linewidths=0, label=ct)

x, y = tcga['GEP_score'].values, tcga['SMA_zscore'].values
m, b, *_ = stats.linregress(x, y)
xr = np.linspace(x.min(), x.max(), 200)
ax_a.plot(xr, m*xr+b, color='#333', lw=1.5, ls='--', alpha=0.8)

pt = f'p = {p_t:.3f}' if p_t >= 0.001 else f'p = {p_t:.2e}'
ax_a.text(0.05, 0.95,
          f'r = {r_t:.2f}, {pt} (ns)\nNo correlation',
          transform=ax_a.transAxes, fontsize=TS, va='top',
          bbox=dict(boxstyle='round,pad=0.35', fc='#F0F0F0', ec='#999', alpha=0.95))

ax_a.set_xlabel('GEP score (Tumor immune activity)', fontsize=LS)
ax_a.set_ylabel('SMA z-score (Host body composition)', fontsize=LS)
ax_a.set_title(f'TCGA (n = {len(tcga)})', fontsize=TIS, fontweight='bold')
ax_a.tick_params(labelsize=TS)
ax_a.spines[['top','right']].set_visible(False)
lg = ax_a.legend(fontsize=7.5, frameon=True, loc='lower right',
                 markerscale=1.2, handletextpad=0.3, labelspacing=0.2)
lg.get_frame().set_edgecolor('#ccc')
ax_a.text(-0.12, 1.05, 'A', transform=ax_a.transAxes,
          fontsize=PS, fontweight='bold')

# ════════════════════════════════════════════════════════════════
# (B) CPTAC-3: GEP vs SMA scatter
# ════════════════════════════════════════════════════════════════

ax_b = fig.add_subplot(gs[0, 1])

for ct, grp in cptac.groupby('cancer_type'):
    ax_b.scatter(grp['GEP_score'], grp['SMA_zscore'],
                 c=CPTAC_COLORS.get(ct, '#888'), alpha=0.6, s=22,
                 linewidths=0, label=ct)

x2, y2 = cptac['GEP_score'].values, cptac['SMA_zscore'].values
m2, b2, *_ = stats.linregress(x2, y2)
xr2 = np.linspace(x2.min(), x2.max(), 200)
ax_b.plot(xr2, m2*xr2+b2, color='#333', lw=1.5, ls='--', alpha=0.8)

pt2 = f'p = {p_c:.3f}' if p_c >= 0.001 else f'p = {p_c:.2e}'
ax_b.text(0.05, 0.95,
          f'r = {r_c:.2f}, {pt2} (ns)\nNo correlation',
          transform=ax_b.transAxes, fontsize=TS, va='top',
          bbox=dict(boxstyle='round,pad=0.35', fc='#F0F0F0', ec='#999', alpha=0.95))

ax_b.set_xlabel('GEP score (Tumor immune activity)', fontsize=LS)
ax_b.set_ylabel('SMA z-score (Host body composition)', fontsize=LS)
ax_b.set_title(f'CPTAC-3 (n = {len(cptac)})', fontsize=TIS, fontweight='bold')
ax_b.tick_params(labelsize=TS)
ax_b.spines[['top','right']].set_visible(False)
lg2 = ax_b.legend(fontsize=8, frameon=True, loc='lower right',
                  markerscale=1.2, handletextpad=0.3, labelspacing=0.2)
lg2.get_frame().set_edgecolor('#ccc')
ax_b.text(-0.12, 1.05, 'B', transform=ax_b.transAxes,
          fontsize=PS, fontweight='bold')

# ════════════════════════════════════════════════════════════════
# (C) Per-cancer-type correlation barplot
# ════════════════════════════════════════════════════════════════

ax_c = fig.add_subplot(gs[1, 0])

records = []
for ct, grp in tcga.groupby('cancer_type'):
    if len(grp) > 5:
        r, p = stats.spearmanr(grp['GEP_score'], grp['SMA_zscore'])
        records.append({'cancer':ct, 'r':r, 'p':p, 'cohort':'TCGA', 'n':len(grp)})
for ct, grp in cptac.groupby('cancer_type'):
    if len(grp) > 5:
        short = {'Uterus, NOS':'UCEC', 'Pancreas':'PDAC', 'Kidney':'CCRCC'}.get(ct, ct)
        r, p = stats.spearmanr(grp['GEP_score'], grp['SMA_zscore'])
        records.append({'cancer':short, 'r':r, 'p':p, 'cohort':'CPTAC-3', 'n':len(grp)})

df_cor = pd.DataFrame(records).sort_values('r')
colors_bar = ['#4C72B0' if c=='TCGA' else '#DD8452' for c in df_cor['cohort']]
ax_c.barh(range(len(df_cor)), df_cor['r'],
          color=colors_bar, alpha=0.82, edgecolor='white', height=0.7)

for i, (_, row) in enumerate(df_cor.iterrows()):
    mk = '*' if row['p'] < 0.05 else 'ns'
    cm = '#c0392b' if row['p'] < 0.05 else '#888'
    xp = row['r'] + (0.01 if row['r'] >= 0 else -0.01)
    ha = 'left' if row['r'] >= 0 else 'right'
    ax_c.text(xp, i, mk, fontsize=8, va='center', ha=ha, color=cm)

ax_c.set_yticks(range(len(df_cor)))
ax_c.set_yticklabels([f"{r['cancer']} (n={r['n']})" for _, r in df_cor.iterrows()],
                     fontsize=TS)
ax_c.axvline(0, color='black', lw=0.8)
ax_c.set_xlabel("Spearman's r (GEP vs SMA)", fontsize=LS)
ax_c.set_title('Correlation by Cancer Type', fontsize=TIS, fontweight='bold')
ax_c.tick_params(axis='x', labelsize=TS)
ax_c.spines[['top','right']].set_visible(False)
ax_c.set_xlim(-0.5, 0.5)

p1 = mpatches.Patch(color='#4C72B0', label='TCGA')
p2 = mpatches.Patch(color='#DD8452', label='CPTAC-3')
lg3 = ax_c.legend(handles=[p1,p2], fontsize=8.5, frameon=True, loc='lower right')
lg3.get_frame().set_edgecolor('#ccc')

# ── 하단 한 줄 추가 ────────────────────────────────────────
ax_c.text(0.5, -0.14,
          'No significant correlation across all cancer types',
          transform=ax_c.transAxes, ha='center', va='top',
          fontsize=9.5, style='italic', color='#555',
          bbox=dict(boxstyle='round,pad=0.3', fc='#F8F8F8', ec='#ccc', alpha=0.9))

ax_c.text(-0.18, 1.05, 'C', transform=ax_c.transAxes,
          fontsize=PS, fontweight='bold')

# ════════════════════════════════════════════════════════════════
# (D) Adjusted Cox forest plot
# ════════════════════════════════════════════════════════════════

ax_d = fig.add_subplot(gs[1, 1])

vars_order = ['GEP Score', 'SMA z-score', 'SMA × GEP']
group_gap  = 1.1
coh_offset = {'TCGA': -0.14, 'CPTAC-3': 0.14}

for vi, var in enumerate(vars_order):
    base_y = vi * group_gap
    if vi > 0:
        ax_d.axhline(base_y - group_gap*0.45, color='#e0e0e0', lw=0.9, zorder=0)
    ax_d.text(0.5, base_y + 0.35, var,
              transform=ax_d.get_yaxis_transform(),
              fontsize=10.5, fontweight='bold', va='bottom', ha='center',
              color='#222')
    for coh in ['TCGA', 'CPTAC-3']:
        row = df_cox[(df_cox['variable']==var) & (df_cox['cohort']==coh)].iloc[0]
        y   = base_y + coh_offset[coh]
        col = coh_colors[coh]
        ax_d.plot([row['lo'], row['hi']], [y, y],
                  color=col, lw=2.2, alpha=0.9, zorder=2)
        ax_d.scatter(row['hr'], y, color=col, s=60, zorder=3,
                     edgecolors='white', linewidths=0.5)
        star  = row['star']
        s_col = '#c0392b' if star != 'ns' else '#888'
        s_fw  = 'bold'    if star != 'ns' else 'normal'
        ax_d.text(row['hi'] + 0.02, y, star,
                  fontsize=9, va='center', ha='left',
                  color=s_col, fontweight=s_fw, zorder=4)

ax_d.axvline(1.0, color='#444', lw=0.9, ls='--', alpha=0.7, zorder=1)
ax_d.set_xlabel('Hazard Ratio (95% CI)', fontsize=LS)
ax_d.set_title('Adjusted Cox Model\n(age, sex, stage)', fontsize=TIS, fontweight='bold')
ax_d.set_xlim(0.2, 2.6)
ax_d.set_ylim(-0.6, len(vars_order)*group_gap - 0.2)
ax_d.set_yticks([])
ax_d.spines[['top','right','left']].set_visible(False)
ax_d.tick_params(axis='x', labelsize=TS)

pt = mpatches.Patch(color='#4C72B0', label='TCGA')
pc = mpatches.Patch(color='#DD8452', label='CPTAC-3')
lg4 = ax_d.legend(handles=[pt,pc], fontsize=9, frameon=True,
                  loc='upper right', title='Cohort', title_fontsize=9)
lg4.get_frame().set_edgecolor('#ccc')

ax_d.text(0.98, 0.02,
          "*** p<0.001\n ** p<0.01\n  * p<0.05\n ns p≥0.05",
          transform=ax_d.transAxes, fontsize=7.5,
          va='bottom', ha='right', family='monospace', color='#555',
          bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.9))

ax_d.text(-0.12, 1.05, 'D', transform=ax_d.transAxes,
          fontsize=PS, fontweight='bold')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 제목 & 저장                          ║
# ╚══════════════════════════════════════════════════════════════╝

fig.suptitle(
    'Independence of Tumor Immune Activity and Host Body Composition',
    fontsize=13.5, fontweight='bold', y=0.97)

plt.savefig(OUT_TIFF, dpi=300, format='tiff',
            bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n저장 완료: {OUT_TIFF}")