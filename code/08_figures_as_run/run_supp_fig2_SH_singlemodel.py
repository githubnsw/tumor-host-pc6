import matplotlib
matplotlib.use('Agg')
"""
Figure 2. Survival Differences Across Immune–SMA Subgroups
============================================================
Panel A: TCGA KM (4 groups)
Panel B: CPTAC-3 KM (4 groups)
Panel C: HR forest — vs IH_SH (adjusted Cox)

수정사항:
  - 전체 글씨 크기 키움
  - Figure 제목: Figure 2. Survival Differences Across Immune–SMA Subgroups
  - A, B: ylim 0.0~1.02 (빈공간 없음), at-risk table 제거
  - A, B 범례: lower left
  - C: TCGA(파란색) 위, CPTAC-3(주황색) 아래 순서

입력:
  TCGA_with_genes.csv
  External_with_genes.csv

출력:
  Figure2.tiff  (dpi=300)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
import warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    파일 경로 설정                              ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_DIR   = r"C:/Users/nsw/AppData/Local/Temp/claude/c--Users-nsw-Desktop-body-composition/61cfca76-a232-4ce7-a356-6d92f663b762/scratchpad/repro_final_v2"
TCGA_FILE  = os.path.join(BASE_DIR, "TCGA_with_genes.csv")
CPTAC_FILE = os.path.join(BASE_DIR, "External_with_genes.csv")
OUT_TIFF   = r'C:/Users/nsw/Desktop/body_composition/revision_r1/final_v2/relabel_SH/SuppFig2_SH_singlemodel.tiff'

# ╔══════════════════════════════════════════════════════════════╗
# ║                    데이터 로드 & 전처리                        ║
# ╚══════════════════════════════════════════════════════════════╝

tcga  = pd.read_csv(TCGA_FILE)
cptac = pd.read_csv(CPTAC_FILE)

def clean_stage(s):  # unified spec: binary stage, missing/unknown -> NaN (excluded)
    s = str(s).upper().replace('STAGE', '').strip()
    if s in ('', 'NAN', 'NONE', 'UNKNOWN', '[DISCREPANCY]', '[NOT AVAILABLE]', '[NOT APPLICABLE]'): return np.nan
    if s.startswith(('III', 'IV')): return 1.0
    if s.startswith(('I', 'II')): return 0.0
    return np.nan

for df in [tcga, cptac]:
    df['stage_num'] = df['stage'].apply(clean_stage)
    df['sex_num']   = (df['sex'].astype(str).str.upper() == 'MALE').astype(float)

GROUP_MAP = {
    'Group4_HighMuscle_HighImmune': 'IH_SH',
    'Group3_HighMuscle_LowImmune' : 'IL_SH',
    'Group2_LowMuscle_HighImmune' : 'IH_SL',
    'Group1_LowMuscle_LowImmune'  : 'IL_SL'
}
GROUP_ORDER  = ['IH_SH', 'IL_SH', 'IH_SL', 'IL_SL']
GROUP_LABELS = {
    'IH_SH': 'IH_SH (Immune-High / SMA-High)',
    'IL_SH': 'IL_SH (Immune-Low / SMA-High)',
    'IH_SL': 'IH_SL (Immune-High / SMA-Low)',
    'IL_SL': 'IL_SL (Immune-Low / SMA-Low)',
}
GROUP_COLORS = {
    'IH_SH': '#2166AC', 'IL_SH': '#74ADD1',
    'IH_SL': '#F46D43', 'IL_SL': '#D73027'
}

for df in [tcga, cptac]:
    df['group'] = df['combined_group'].map(GROUP_MAP)

def pstar(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Adjusted Cox (vs IH_SH)                   ║
# ╚══════════════════════════════════════════════════════════════╝

def cox_vs_ref(df, ref='IH_SH'):
    results = {}
    df2 = df[['OS_time','OS','group','age','sex_num','stage_num','cancer_type']].dropna()
    df2 = df2[df2['OS_time'] > 0].copy()
    gs_ = [g for g in GROUP_ORDER if g != ref]
    for g in gs_: df2[g] = (df2['group'] == g).astype(float)
    covs = ['OS_time','OS'] + gs_ + ['age','stage_num','cancer_type'] + (['sex_num'] if df2['sex_num'].std() > 0.05 else [])
    cph1 = CoxPHFitter(); cph1.fit(df2[covs].dropna(), duration_col='OS_time', event_col='OS', strata='cancer_type')  # single 4-level model
    for g in gs_:
        try:
            s = cph1.summary.loc[g]
            results[g] = {'hr':s['exp(coef)'], 'lo':s['exp(coef) lower 95%'],
                          'hi':s['exp(coef) upper 95%'], 'p':s['p'], 'star':pstar(s['p'])}
        except:
            results[g] = {'hr':np.nan, 'lo':np.nan, 'hi':np.nan, 'p':np.nan, 'star':'na'}
    return results

print("Cox TCGA...")
tcga_p  = multivariate_logrank_test(tcga['OS_time'], tcga['group'], tcga['OS']).p_value
tcga_hr = cox_vs_ref(tcga)

print("Cox CPTAC...")
cptac_p  = multivariate_logrank_test(cptac['OS_time'], cptac['group'], cptac['OS']).p_value
cptac_hr = cox_vs_ref(cptac)

def _stratified_logrank(df, T='OS_time', E='OS', G='group', S_='cancer_type'):  # same implementation as r2_8 / fig4_final
    from scipy.stats import chi2 as _chi2
    d0 = df[[T, E, G, S_]].dropna(); d0 = d0[d0[T] > 0]
    groups = sorted(d0[G].unique()); k = len(groups); OE = np.zeros(k); V = np.zeros((k, k))
    for _, d in d0.groupby(S_):
        for t in np.sort(d.loc[d[E] == 1, T].unique()):
            at = d[d[T] >= t]; n = len(at); dead = d[(d[T] == t) & (d[E] == 1)]; dt = len(dead)
            if n < 2 or dt == 0: continue
            nj = np.array([(at[G] == g).sum() for g in groups], float); dj = np.array([(dead[G] == g).sum() for g in groups], float)
            p = nj / n; OE += dj - dt * p; V += dt * (n - dt) / (n - 1) * (np.diag(p) - np.outer(p, p))
    oe, v = OE[:-1], V[:-1, :-1]; stat = float(oe @ np.linalg.pinv(v) @ oe); return float(_chi2.sf(stat, k - 1))
tcga_p_strat = _stratified_logrank(tcga); cptac_p_strat = _stratified_logrank(cptac)
STRAT_P = {'A': tcga_p_strat, 'B': cptac_p_strat}
print(f"stratified log-rank: TCGA p={tcga_p_strat:.4g}, CPTAC p={cptac_p_strat:.4g}")

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 레이아웃                             ║
# ╚══════════════════════════════════════════════════════════════╝

fig = plt.figure(figsize=(20, 8))
fig.patch.set_facecolor('white')
gs = GridSpec(1, 3, figure=fig,
              wspace=0.40, left=0.06, right=0.97,
              top=0.86, bottom=0.13,
              width_ratios=[1, 1, 0.85])

# ── 글씨 크기 (전체 키움) ──────────────────────────────────
LS, TS, TIS, PS = 14, 12, 15, 17

# ── KM helper ─────────────────────────────────────────────
def draw_km(ax, df, title, panel_label, overall_p, legend_loc='lower left'):
    df2 = df[['OS_time','OS','group']].dropna()
    df2 = df2[df2['OS_time'] > 0].copy()
    df2['OS_months'] = df2['OS_time'] / 30.44

    kmf = KaplanMeierFitter()
    for g in GROUP_ORDER:
        sub = df2[df2['group'] == g]
        kmf.fit(sub['OS_months'], sub['OS'], label=g)
        kmf.plot_survival_function(
            ax=ax, color=GROUP_COLORS[g], lw=2,
            ci_show=True, ci_alpha=0.10,
            show_censors=True,
            censor_styles={'ms': 5, 'marker': '+'})

    p_str = (f'Log-rank p = {overall_p:.2e}'
             if overall_p < 0.001 else f'Log-rank p = {overall_p:.4f}')
    sp = STRAT_P[panel_label]; p_str += (f'\nStratified log-rank p = {sp:.2e}' if sp < 0.001 else f'\nStratified log-rank p = {sp:.4f}')
    ax.text(0.97, 0.97, p_str, transform=ax.transAxes,
            fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))

    ax.set_xlabel('Time (months)', fontsize=LS)
    ax.set_ylabel('Overall Survival Probability', fontsize=LS)
    ax.set_title(title, fontsize=TIS, fontweight='bold')
    ax.set_ylim(0.0, 1.02)   # 빈공간 없애기
    ax.set_xlim(-2, None)
    ax.tick_params(labelsize=TS)
    ax.spines[['top','right']].set_visible(False)
    ax.text(-0.10, 1.05, panel_label, transform=ax.transAxes,
            fontsize=PS, fontweight='bold')

    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
               for g in GROUP_ORDER]
    leg = ax.legend(handles=handles, fontsize=10, frameon=True, loc=legend_loc)
    leg.get_frame().set_edgecolor('#ccc')

# ── A: TCGA ───────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
draw_km(ax_a, tcga, f'TCGA (n = {len(tcga)})', 'A', tcga_p,
        legend_loc='lower left')

# ── B: CPTAC-3 ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
draw_km(ax_b, cptac, f'CPTAC-3 (n = {len(cptac)})', 'B', cptac_p,
        legend_loc='lower left')

# ── C: HR forest — TCGA 위(+0.14), CPTAC-3 아래(-0.14) ───
ax_c = fig.add_subplot(gs[0, 2])

groups_c   = ['IL_SH', 'IH_SL', 'IL_SL']
# TCGA offset +0.14 (위), CPTAC-3 offset -0.14 (아래)
coh_pairs  = [('TCGA',    tcga_hr,  '#4C72B0',  +0.14),
              ('CPTAC-3', cptac_hr, '#DD8452',  -0.14)]
group_gap  = 1.0

for vi, g in enumerate(groups_c):
    base_y = vi * group_gap
    if vi > 0:
        ax_c.axhline(base_y - group_gap*0.45, color='#e0e0e0', lw=0.9, zorder=0)

    ax_c.text(0.5, base_y + 0.32, f'vs IH_SH\n({g})',
              transform=ax_c.get_yaxis_transform(),
              fontsize=11.5, fontweight='bold', va='bottom', ha='center',
              color=GROUP_COLORS[g])

    for coh, hr_dict, col, offset in coh_pairs:
        row = hr_dict[g]
        y   = base_y + offset
        if not np.isnan(row['hr']):
            ax_c.plot([row['lo'], row['hi']], [y, y],
                      color=col, lw=2.4, alpha=0.9, zorder=2)
            ax_c.scatter(row['hr'], y, color=col, s=70, zorder=3,
                         edgecolors='white', linewidths=0.5)
            star  = row['star']
            s_col = '#c0392b' if star != 'ns' else '#888'
            s_fw  = 'bold' if star != 'ns' else 'normal'
            ax_c.text(row['hi'] + 0.05, y, star,
                      fontsize=11.5, va='center', ha='left',
                      color=s_col, fontweight=s_fw, zorder=4)

ax_c.axvline(1.0, color='#444', lw=0.9, ls='--', alpha=0.7, zorder=1)
ax_c.set_xlabel('Hazard Ratio (95% CI)\nvs IH_SH (reference)', fontsize=LS)
ax_c.set_title('Adjusted HR Summary\n(age, sex, stage I/II vs III/IV | strat: cancer type)',
               fontsize=12.5, fontweight='bold')
ax_c.set_xlim(0.2, 6.0)
ax_c.set_ylim(-0.55, len(groups_c)*group_gap - 0.2)
ax_c.set_yticks([])
ax_c.spines[['top','right','left']].set_visible(False)
ax_c.tick_params(axis='x', labelsize=TS)

# 범례: TCGA 먼저
pt       = mpatches.Patch(color='#4C72B0', label='TCGA')
pc_patch = mpatches.Patch(color='#DD8452', label='CPTAC-3')
lg = ax_c.legend(handles=[pt, pc_patch], fontsize=11, frameon=True,
                 loc='upper right', title='Cohort', title_fontsize=11)
lg.get_frame().set_edgecolor('#ccc')

ax_c.text(0.98, 0.02,
          "*** p<0.001\n ** p<0.01\n  * p<0.05\n ns p≥0.05",
          transform=ax_c.transAxes, fontsize=10,
          va='bottom', ha='right', family='monospace', color='#555',
          bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.9))

ax_c.text(-0.10, 1.05, 'C', transform=ax_c.transAxes,
          fontsize=PS, fontweight='bold')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 제목 & 저장                          ║
# ╚══════════════════════════════════════════════════════════════╝

fig.suptitle('Survival Differences Across Immune–SMA Subgroups',
             fontsize=17, fontweight='bold', y=1.01)

plt.savefig(OUT_TIFF, dpi=300, format='tiff',
            bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n저장 완료: {OUT_TIFF}")
# ---- numbers out (pairwise models as in the figure + single 4-level model as sensitivity) ----
rows = []
for coh, df, hrd, pp, ps in [('TCGA', tcga, tcga_hr, tcga_p, tcga_p_strat), ('CPTAC-3', cptac, cptac_hr, cptac_p, cptac_p_strat)]:
    for g, r in hrd.items(): rows.append(dict(cohort=coh, model='pairwise vs IH_SH (figure)', group=g, hr=r['hr'], lo=r['lo'], hi=r['hi'], p=r['p']))
    d2 = df[['OS_time','OS','group','age','sex_num','stage_num','cancer_type']].dropna(); d2 = d2[d2.OS_time > 0].copy()
    for g in ['IL_SH','IH_SL','IL_SL']: d2[g] = (d2.group == g).astype(float)
    covs = ['IL_SH','IH_SL','IL_SL','age','stage_num'] + (['sex_num'] if d2.sex_num.std() > 0.05 else [])
    m = CoxPHFitter().fit(d2[['OS_time','OS','cancer_type'] + covs], 'OS_time', 'OS', strata='cancer_type')
    for g in ['IL_SH','IH_SL','IL_SL']:
        s = m.summary.loc[g]; rows.append(dict(cohort=coh, model='single 4-level model (sensitivity)', group=g, hr=s['exp(coef)'], lo=s['exp(coef) lower 95%'], hi=s['exp(coef) upper 95%'], p=s['p']))
    rows.append(dict(cohort=coh, model='log-rank', group='4 groups pooled', hr=np.nan, lo=np.nan, hi=np.nan, p=pp)); rows.append(dict(cohort=coh, model='log-rank', group='4 groups stratified', hr=np.nan, lo=np.nan, hi=np.nan, p=ps))
    rows.append(dict(cohort=coh, model='n', group='analytic n / events', hr=len(d2), lo=int(d2.OS.sum()), hi=np.nan, p=np.nan))
    print(coh, 'group sizes:', df.group.value_counts().to_dict())
R = pd.DataFrame(rows); R.to_csv(r'C:/Users/nsw/Desktop/body_composition/revision_r1/final_v2/relabel_SH/supp_fig2_numbers_relabeled_singlemodel.csv', index=False); print(R.round(4).to_string())
