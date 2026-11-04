import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')
import os

OUTDIR = r"C:\Users\nsw\Desktop\body_composition\pc6_distribution"
os.makedirs(OUTDIR, exist_ok=True)

tcga_pc   = pd.read_csv(r"C:\Users\nsw\Desktop\body_composition\ae_v3_outputs\ae_latent_pca10_all.csv")
cptac_pc  = pd.read_csv(r"C:\Users\nsw\Desktop\body_composition\ae_v3_outputs\ae_latent_pca10_external.csv")
tcga_clin = pd.read_csv(r"C:\Users\nsw\Desktop\body_composition\TCGA_with_genes.csv")
cptac_clin= pd.read_csv(r"C:\Users\nsw\Desktop\body_composition\External_with_genes.csv")

tcga_pc['pid']    = tcga_pc['patient_id'].str[:12]
cptac_pc['pid']   = cptac_pc['patient_id'].str[:12]
tcga_clin['pid']  = tcga_clin['patient_id'].str[:12]
cptac_clin['pid'] = cptac_clin['patient_id'].str[:12]

tcga  = tcga_pc.merge(tcga_clin[['pid','cancer_type']], on='pid', how='left')
cptac = cptac_pc.merge(cptac_clin[['pid','cancer_type']], on='pid', how='left')
cptac['cancer_type'] = cptac['cancer_type'].map(
    {'Uterus, NOS':'UCEC','Kidney':'CCRCC','Pancreas':'PDAC'})

pc6_tcga  = tcga['AE_PC6'].dropna().values
pc6_cptac = cptac['AE_PC6'].dropna().values

pc6_tcga_z  = (pc6_tcga  - pc6_tcga.mean())  / pc6_tcga.std()
pc6_cptac_z = (pc6_cptac - pc6_cptac.mean()) / pc6_cptac.std()

ks_stat, ks_p = stats.ks_2samp(pc6_tcga_z, pc6_cptac_z)
x_range = np.linspace(-4, 4, 500)
kde_t = gaussian_kde(pc6_tcga_z)(x_range)
kde_c = gaussian_kde(pc6_cptac_z)(x_range)
ovl = np.trapz(np.minimum(kde_t, kde_c), x_range)

tcga['pc6_z']  = (tcga['AE_PC6']  - pc6_tcga.mean())  / pc6_tcga.std()
cptac['pc6_z'] = (cptac['AE_PC6'] - pc6_cptac.mean()) / pc6_cptac.std()

common_cts = {
    'UCEC': ('UCEC', 'UCEC'),
    'Kidney\n(KIRC / CCRCC)': ('KIRC', 'CCRCC'),
}
ct_comparison = {}
for label, (tc, cp) in common_cts.items():
    t_vals = tcga[tcga['cancer_type']==tc]['pc6_z'].dropna().values
    c_vals = cptac[cptac['cancer_type']==cp]['pc6_z'].dropna().values
    ks_s, ks_p2 = stats.ks_2samp(t_vals, c_vals)
    ct_comparison[label] = {
        'tcga': t_vals, 'cptac': c_vals,
        'ks_stat': ks_s, 'ks_p': ks_p2,
        'n_tcga': len(t_vals), 'n_cptac': len(c_vals),
    }

COL_TCGA  = '#2874A6'
COL_CPTAC = '#C0392B'

# ── Figure: 2행 레이아웃 (Panel A 전체 1행, C/D/E 2행) ───
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')

# suptitle 공간 확보를 위해 top 낮춤
gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.45, wspace=0.32,
                       left=0.07, right=0.97,
                       top=0.88, bottom=0.08)  # top=0.88로 공간 확보

# suptitle을 충분히 위에 배치
fig.suptitle(
    'PC6 Score Distribution: Cross-Cohort Comparability\n'
    'TCGA-Trained Autoencoder + PCA Applied to CPTAC-3',
    fontsize=12, fontweight='bold', color='#1A3A6B',
    y=0.97  # 높게 배치
)

# ── Panel A: 전체 1행 차지 ────────────────────────────────
ax_a = fig.add_subplot(gs[0, :])  # 3컬럼 모두 사용

ax_a.fill_between(x_range, kde_t, alpha=0.65,
                  color=COL_TCGA,  label=f'TCGA (n={len(pc6_tcga_z)})')
ax_a.fill_between(x_range, kde_c, alpha=0.65,
                  color=COL_CPTAC, label=f'CPTAC-3 (n={len(pc6_cptac_z)})')
ax_a.fill_between(x_range, np.minimum(kde_t, kde_c),
                  alpha=0.30, color='#8E44AD',
                  label=f'Overlap (OVL={ovl:.2f})')
ax_a.axvline(0, color='#888888', lw=1.0, linestyle=':', alpha=0.6)

ax_a.set_xlabel('PC6 Score (within-cohort z-standardized)', fontsize=10)
ax_a.set_ylabel('Density', fontsize=10)

# Panel 제목은 왼쪽 정렬 + 작게
ax_a.set_title(
    'A  Overall PC6 Score Distribution: TCGA vs. CPTAC-3'
    '   (within-cohort z-standardized)',
    fontsize=10, fontweight='bold', color='#1A3A6B',
    loc='left', pad=8
)

ax_a.legend(fontsize=9, framealpha=0.85, loc='upper left')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.grid(alpha=0.2)

ks_str = 'p<0.001' if ks_p < 0.001 else f'p={ks_p:.3f}'
ax_a.text(0.98, 0.97,
          f'KS test: D={ks_stat:.3f}, {ks_str}\nOVL={ovl:.2f}',
          transform=ax_a.transAxes, ha='right', va='top',
          fontsize=9, color='#555555',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#CCCCCC', alpha=0.9))

# SD 정보를 그래프 내부에 텍스트로 표시 (Panel B 대체)
ax_a.text(0.98, 0.78,
          f'Within-cohort SD:\nTCGA: {pc6_tcga.std():.2f}  |  CPTAC-3: {pc6_cptac.std():.2f}',
          transform=ax_a.transAxes, ha='right', va='top',
          fontsize=8.5, color='#444444',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F8F8',
                    edgecolor='#CCCCCC', alpha=0.9))

# ── Panel C & D: 공통 암종 ────────────────────────────────
for col_idx, (label, res) in enumerate(ct_comparison.items()):
    ax = fig.add_subplot(gs[1, col_idx])

    t_vals = res['tcga']
    c_vals = res['cptac']
    all_vals = np.concatenate([t_vals, c_vals])
    x_ct = np.linspace(all_vals.min()-0.5, all_vals.max()+0.5, 300)

    kde_ct_t = gaussian_kde(t_vals)(x_ct)
    kde_ct_c = gaussian_kde(c_vals)(x_ct)

    ax.fill_between(x_ct, kde_ct_t, alpha=0.65, color=COL_TCGA,
                    label=f"TCGA (n={res['n_tcga']})")
    ax.fill_between(x_ct, kde_ct_c, alpha=0.65, color=COL_CPTAC,
                    label=f"CPTAC-3 (n={res['n_cptac']})")
    ax.axvline(np.median(t_vals), color=COL_TCGA, lw=1.5, ls='--', alpha=0.7)
    ax.axvline(np.median(c_vals), color=COL_CPTAC, lw=1.5, ls='--', alpha=0.7)

    ks_str2 = 'p<0.001' if res['ks_p'] < 0.001 else f"p={res['ks_p']:.3f}"
    panel_letter = chr(66 + col_idx)  # B, C
    ax.set_title(f'{panel_letter}  {label}\n(z-standardized within cohort)',
                 fontsize=10, fontweight='bold', color='#1A3A6B', loc='left')
    ax.set_xlabel('PC6 Score (z-standardized)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8.5, framealpha=0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2)
    ax.text(0.98, 0.97,
            f'KS: D={res["ks_stat"]:.3f}, {ks_str2}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8.5, color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

# ── Panel E → D: TCGA 암종별 box ─────────────────────────
ax_e = fig.add_subplot(gs[1, 2])

ct_order = (tcga.groupby('cancer_type')['pc6_z']
               .median().sort_values(ascending=False).index.tolist())
ct_data_list = [tcga[tcga['cancer_type']==ct]['pc6_z'].dropna().values
                for ct in ct_order]
ct_ns = [len(d) for d in ct_data_list]

bp = ax_e.boxplot(ct_data_list, positions=range(len(ct_order)),
                  patch_artist=True, widths=0.55, vert=True,
                  medianprops=dict(color='#333333', linewidth=2),
                  whiskerprops=dict(color='#888888'),
                  capprops=dict(color='#888888'),
                  flierprops=dict(marker='o', markersize=2.5,
                                  alpha=0.4, color='#AAAAAA'))
for patch in bp['boxes']:
    patch.set_facecolor(COL_TCGA)
    patch.set_alpha(0.65)
    patch.set_edgecolor('white')

cptac_q25z = np.percentile(pc6_cptac_z, 25)
cptac_q75z = np.percentile(pc6_cptac_z, 75)
ax_e.axhspan(cptac_q25z, cptac_q75z, alpha=0.15,
             color=COL_CPTAC, label='CPTAC-3 IQR')
ax_e.axhline(0, color=COL_CPTAC, lw=1.5, linestyle='--', alpha=0.7,
             label='CPTAC-3 median')

ax_e.set_xticks(range(len(ct_order)))
ax_e.set_xticklabels(
    [f'{ct}\n(n={n})' for ct, n in zip(ct_order, ct_ns)],
    fontsize=7.5, rotation=30, ha='right')
ax_e.set_ylabel('PC6 Score (z-standardized)', fontsize=9)
ax_e.set_title('D  TCGA Cancer Type–Specific PC6\n(vs. CPTAC-3 overall)',
               fontsize=10, fontweight='bold', color='#1A3A6B', loc='left')
ax_e.legend(fontsize=8, loc='upper right', framealpha=0.85)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)
ax_e.grid(axis='y', alpha=0.2)

out_path = f"{OUTDIR}/supp_figure_pc6_distribution.tiff"
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"저장: {out_path}")