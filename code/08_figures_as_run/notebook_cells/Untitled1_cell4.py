"""
Figure 4. Clinical Impact of PC6 in Combination with Body Composition
======================================================================
Panel A: TCGA multivariable Cox (SMA + PC6 forest plot)
Panel B: CPTAC-3 multivariable Cox (동일 구조)
Panel C: Risk surface (SMA × PC6, TCGA)
         - y축: -3~3 고정
         - x축: 데이터 범위 (그림 중앙 배치)
         - Highest risk 박스: 왼쪽 상단
         - Relative hazard 박스: 오른쪽 상단
         - Lowest risk 박스: 우하단

입력:
  TCGA_with_genes.csv
  External_with_genes.csv
  ae_latent_pca10_all.csv
  ae_latent_pca10_external.csv

출력:
  Figure4.tiff  (dpi=300)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    파일 경로 설정                              ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_DIR  = r"C:\Users\nsw\Desktop\body_composition"
TCGA_FILE = os.path.join(BASE_DIR, "TCGA_with_genes.csv")
CPTAC_FILE= os.path.join(BASE_DIR, "External_with_genes.csv")
AE_TCGA   = os.path.join(BASE_DIR, "ae_latent_pca10_all.csv")
AE_CPTAC  = os.path.join(BASE_DIR, "ae_latent_pca10_external.csv")
OUT_TIFF  = os.path.join(BASE_DIR, "Figure4.tiff")

# ╔══════════════════════════════════════════════════════════════╗
# ║                    데이터 로드 & 전처리                        ║
# ╚══════════════════════════════════════════════════════════════╝

tcga     = pd.read_csv(TCGA_FILE)
cptac    = pd.read_csv(CPTAC_FILE)
pc_tcga  = pd.read_csv(AE_TCGA)
pc_cptac = pd.read_csv(AE_CPTAC)

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
    df['sex_num']   = (df['sex'].astype(str).str.upper() == 'MALE').astype(float)

tcga['pid']      = tcga['patient_id'].astype(str).str[:12]
pc_tcga['pid']   = pc_tcga['patient_id'].astype(str).str[:12]
cptac['pid']     = cptac['patient_id'].astype(str)
pc_cptac['pid']  = pc_cptac['patient_id'].astype(str)

m_tcga  = pc_tcga.merge(
    tcga[['pid','OS_time','OS','cancer_type','age','stage_num','sex_num','SMA_zscore']],
    on='pid')
m_cptac = pc_cptac.merge(
    cptac[['pid','OS_time','OS','cancer_type','age','stage_num','sex_num','SMA_zscore']],
    on='pid')

for df in [m_tcga, m_cptac]:
    df['PC6_z'] = (df['AE_PC6'] - df['AE_PC6'].mean()) / df['AE_PC6'].std()

def pstar(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ╔══════════════════════════════════════════════════════════════╗
# ║          Cox: SMA + PC6, adj age/sex/stage, strat cancer     ║
# ╚══════════════════════════════════════════════════════════════╝

def run_cox(df):
    covs = ['SMA_zscore', 'PC6_z', 'age', 'stage_num']
    if df['sex_num'].std() > 0.05:
        covs.append('sex_num')
    d = df[['OS_time','OS','cancer_type'] + covs].dropna()
    d = d[d['OS_time'] > 0].copy()
    cph = CoxPHFitter()
    cph.fit(d, duration_col='OS_time', event_col='OS',
            strata='cancer_type', robust=True)
    return cph.summary, d, cph

print("Fitting Cox models...")
res_t, d_t, cph_t = run_cox(m_tcga)
res_c, d_c, cph_c = run_cox(m_cptac)

print(f"TCGA  n={len(d_t)}, events={int(d_t['OS'].sum())}")
print(f"CPTAC n={len(d_c)}, events={int(d_c['OS'].sum())}")

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 레이아웃                             ║
# ╚══════════════════════════════════════════════════════════════╝

LS, TS, TIS, PS = 11, 9.5, 12, 14
VAR_COLORS = {'SMA_zscore': '#2166AC', 'PC6_z': '#D73027'}
VAR_LABELS = {'SMA_zscore': 'SMA z-score', 'PC6_z': 'PC6 (z-score)'}
VAR_MK     = {'SMA_zscore': 'D', 'PC6_z': 's'}

fig = plt.figure(figsize=(20, 7))
fig.patch.set_facecolor('white')
gs = GridSpec(1, 3, figure=fig,
              wspace=0.38, left=0.05, right=0.97,
              top=0.90, bottom=0.10,
              width_ratios=[1, 1, 1.2])

# ════════════════════════════════════════════════════════════════
# Forest plot helper (A, B 공통)
# ════════════════════════════════════════════════════════════════

def draw_forest(ax, summary, cohort, panel_label, n, events):
    X_MIN = 0.25; X_MAX = 2.30
    X_CI_START = 0.55    # CI 시작 x (라벨 영역과 분리)
    LABEL_X = 0.28       # 라벨/텍스트 x 위치

    y_pos = {'SMA_zscore': 2.0, 'PC6_z': 0.0}

    ax.axvline(1.0, color='#888', lw=1.0, ls='--', alpha=0.6, zorder=1)
    ax.axhline(1.0, color='#e0e0e0', lw=0.9, zorder=0)

    for var in ['SMA_zscore', 'PC6_z']:
        r      = summary.loc[var]
        hr     = r['exp(coef)']
        lo     = r['exp(coef) lower 95%']
        hi     = r['exp(coef) upper 95%']
        p      = r['p']
        y      = y_pos[var]
        col    = VAR_COLORS[var]
        is_sig = (p < 0.05)

        # 배경 shading
        ax.axhspan(y - 0.9, y + 0.9, color=col, alpha=0.05, zorder=0)

        # variable 라벨 — 왼쪽 고정
        ax.text(LABEL_X, y + 0.35, VAR_LABELS[var],
                fontsize=11, va='center', ha='left',
                fontweight='bold', color=col)

        # CI line (라벨 영역 이후부터)
        lo_plot = max(lo, X_CI_START)
        ax.plot([lo_plot, hi], [y, y], color=col, lw=2.8, alpha=0.9, zorder=2)

        # point (SMA=다이아몬드, PC6=사각형)
        mk = VAR_MK[var]
        ax.scatter(hr, y, color=col, s=120, marker=mk, zorder=3,
                   edgecolors='white', linewidths=0.8)

        # HR 텍스트 — 라벨 아래 (겹침 방지)
        ax.text(LABEL_X, y - 0.20,
                f"HR={hr:.3f} ({lo:.3f}–{hi:.3f})",
                fontsize=8.5, va='top', ha='left',
                color=col if is_sig else '#999',
                fontweight='bold' if is_sig else 'normal')
        ax.text(LABEL_X, y - 0.62,
                f"p={p:.4f} {pstar(p)}",
                fontsize=8.5, va='top', ha='left',
                color=col if is_sig else '#999',
                fontweight='bold' if is_sig else 'normal')

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.9, 3.0)
    ax.set_yticks([])
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=LS)
    ax.set_title(f'({panel_label}) {cohort}: multivariable Cox model\n'
                 f'n={n}, events={events} | adj: age, sex, stage | strat: cancer type',
                 fontsize=10.5, fontweight='bold')
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='x', labelsize=TS)
    ax.text(-0.10, 1.06, panel_label, transform=ax.transAxes,
            fontsize=PS, fontweight='bold')

    h1 = Line2D([0],[0], marker='D', color='w', markerfacecolor='#2166AC',
                markeredgecolor='white', markersize=9, label='SMA z-score')
    h2 = Line2D([0],[0], marker='s', color='w', markerfacecolor='#D73027',
                markeredgecolor='white', markersize=9, label='PC6')
    ax.legend(handles=[h1, h2], fontsize=9, frameon=True,
              loc='lower right').get_frame().set_edgecolor('#ccc')

# ── A: TCGA ───────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
draw_forest(ax_a, res_t, 'TCGA', 'A', len(d_t), int(d_t['OS'].sum()))

# ── B: CPTAC-3 ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
draw_forest(ax_b, res_c, 'CPTAC-3', 'B', len(d_c), int(d_c['OS'].sum()))

# ════════════════════════════════════════════════════════════════
# Panel C: Risk surface
# ════════════════════════════════════════════════════════════════

ax_c = fig.add_subplot(gs[0, 2])

covs_g = ['SMA_zscore', 'PC6_z', 'age', 'stage_num']
if m_tcga['sex_num'].std() > 0.05:
    covs_g.append('sex_num')

# x: 데이터 범위, y: -3~3 고정
sma_q01 = m_tcga['SMA_zscore'].quantile(0.01)
sma_q99 = m_tcga['SMA_zscore'].quantile(0.99)

sma_grid = np.linspace(sma_q01, sma_q99, 80)
pc6_grid = np.linspace(-3, 3, 80)
SMA_G, PC6_G = np.meshgrid(sma_grid, pc6_grid)

grid_df = pd.DataFrame({
    'SMA_zscore': SMA_G.ravel(),
    'PC6_z'     : PC6_G.ravel(),
    'age'       : d_t['age'].median(),
    'stage_num' : d_t['stage_num'].median(),
})
if 'sex_num' in covs_g:
    grid_df['sex_num'] = d_t['sex_num'].median()

params = cph_t.params_
lp = grid_df[covs_g].values @ params[covs_g].values
RH = np.exp(lp.reshape(SMA_G.shape))

# filled contour
im = ax_c.contourf(SMA_G, PC6_G, RH, levels=30, cmap='RdYlBu_r', alpha=0.92)

# white contour lines + labels
cs = ax_c.contour(SMA_G, PC6_G, RH, levels=8,
                   colors='white', linewidths=0.8, alpha=0.7)
ax_c.clabel(cs, inline=True, fontsize=7.5, fmt='%.2f', colors='white')

# colorbar
cb = fig.colorbar(im, ax=ax_c, shrink=0.85, pad=0.02)
cb.set_label('Relative hazard', fontsize=9)
cb.ax.tick_params(labelsize=8)

# patient scatter
ax_c.scatter(m_tcga['SMA_zscore'], m_tcga['PC6_z'],
             c='white', s=8, alpha=0.35, linewidths=0, zorder=3)

# reference lines
ax_c.axvline(0, color='#333', lw=1.2, ls='--', alpha=0.8)
ax_c.axhline(0, color='#333', lw=1.2, ls='--', alpha=0.8)

# 축 범위 설정
ax_c.set_xlim(sma_q01 - 0.3, sma_q99 + 0.3)
ax_c.set_ylim(-3, 3)

# ── 텍스트 박스 배치 (data 좌표계) ────────────────────────
# 1) Highest risk — 왼쪽 상단 (그림 안, 겹침 없음)
ax_c.text(sma_q01, 2.75,
          'Highest risk\nLow SMA\nHigh PC6',
          ha='left', va='top',
          fontsize=8, color='white', fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.3', fc='#8B0000', alpha=0.88),
          zorder=5)

# 2) Relative hazard — 오른쪽으로 밀어서 (SMA=0.3)
ax_c.text(0.3, 2.75,
          'Relative hazard (TCGA Cox model)\nRR = exp(β_SMA·SMA_z + β_PC6·PC6_z)',
          ha='left', va='top',
          fontsize=7.5, color='#222',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbb', alpha=0.92),
          zorder=5)

# 3) Lowest risk — 우하단
ax_c.text(0.98, 0.03,
          'Lowest risk\nHigh SMA\nLow PC6',
          transform=ax_c.transAxes, ha='right', va='bottom',
          fontsize=8, color='white', fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.25', fc='#08519C', alpha=0.85),
          zorder=5)

ax_c.set_xlabel('SMA z-score', fontsize=LS)
ax_c.set_ylabel('PC6 z-score', fontsize=LS)
ax_c.set_title('Risk Surface: SMA × PC6\n(TCGA-centered joint model)',
               fontsize=TIS, fontweight='bold')
ax_c.tick_params(labelsize=TS)
ax_c.spines[['top','right']].set_visible(False)
ax_c.text(-0.12, 1.06, 'C', transform=ax_c.transAxes,
          fontsize=PS, fontweight='bold')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 제목 & 저장                          ║
# ╚══════════════════════════════════════════════════════════════╝

fig.suptitle(
    'Clinical Impact of PC6 in Combination with Body Composition',
    fontsize=11.5, fontweight='bold', y=1.01)

plt.savefig(OUT_TIFF, dpi=300, format='tiff',
            bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n저장 완료: {OUT_TIFF}")