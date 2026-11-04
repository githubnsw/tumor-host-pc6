"""
Figure 5. PC6 Represents a Conserved Warburg-like Metabolic Program
=====================================================================
Layout (1 row × 2 panels):
  A: Pathway-level mean r (TCGA vs CPTAC-3)
  B: ssGSEA NES ~ PC6     (TCGA vs CPTAC-3)

(원래 A, B [gene-level barplot]은 Supplementary로 분리됨)
(이전 C, D [Warburg scatter]도 제거 — 두 cohort에서 동일 결과 재확인 후)

입력:
  RNA_Group1~4_updated_to_new_symbol.csv  (TCGA)
  ex_RNA_Group1~4.csv                     (CPTAC-3)
  ae_latent_pca10_all.csv
  ae_latent_pca10_external.csv

출력:
  Figure5.tiff
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    파일 경로 설정                              ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_DIR = r"C:\Users\nsw\Desktop\body_composition"

TCGA_RNA_FILES = [
    os.path.join(BASE_DIR, "RNA_Group1_updated_to_new_symbol.csv"),
    os.path.join(BASE_DIR, "RNA_Group2_updated_to_new_symbol.csv"),
    os.path.join(BASE_DIR, "RNA_Group3_updated_to_new_symbol.csv"),
    os.path.join(BASE_DIR, "RNA_Group4_updated_to_new_symbol.csv"),
]
CPTAC_RNA_FILES = [
    os.path.join(BASE_DIR, "ex_RNA_Group1_LowMuscle_LowImmune.csv"),
    os.path.join(BASE_DIR, "ex_RNA_Group2_LowMuscle_HighImmune.csv"),
    os.path.join(BASE_DIR, "ex_RNA_Group3_HighMuscle_LowImmune.csv"),
    os.path.join(BASE_DIR, "ex_RNA_Group4_HighMuscle_HighImmune.csv"),
]
TCGA_PC_FILE  = os.path.join(BASE_DIR, "ae_latent_pca10_all.csv")
CPTAC_PC_FILE = os.path.join(BASE_DIR, "ae_latent_pca10_external.csv")
OUT_TIFF      = os.path.join(BASE_DIR, "Figure5.tiff")
OUT_SSGSEA    = os.path.join(BASE_DIR, "results_ssgsea_pc6.csv")

# ╔══════════════════════════════════════════════════════════════╗
# ║                    유전자 세트 정의                             ║
# ╚══════════════════════════════════════════════════════════════╝

GENE_SETS = {
    'Glycolysis'   : ['HK2','PFKP','ALDOA','ENO1','PGK1','PKM','LDHA','GAPDH','TPI1','PGAM1'],
    'Hypoxia'      : ['VEGFA','NDRG1','CA9','HIF1A','ADM','HMOX1'],
    'Lactate/MCT'  : ['LDHA','LDHB','SLC16A3','SLC16A1'],
    'OXPHOS'       : ['SDHA','SDHB','COX5A','NDUFB3','UQCRC1'],
    'Proliferation': ['E2F1','CDK4','CCND1','CDK1','CCNA2','TOP2A','MKI67'],
}

SET_COLORS = {
    'Glycolysis'   : '#E74C3C',
    'Hypoxia'      : '#E67E22',
    'Lactate/MCT'  : '#C0392B',
    'OXPHOS'       : '#2980B9',
    'Proliferation': '#7F8C8D',
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                    데이터 로드 & 전처리                        ║
# ╚══════════════════════════════════════════════════════════════╝

def merge_rna_groups(files):
    dfs = [pd.read_csv(f).set_index('gene') for f in files]
    merged = pd.concat(dfs, axis=1)
    return merged.loc[~merged.index.duplicated(keep='first')]

print("Loading RNA data...")
rna_tcga  = merge_rna_groups(TCGA_RNA_FILES)
rna_cptac = merge_rna_groups(CPTAC_RNA_FILES)
rna_tcga.columns = [c[:12] for c in rna_tcga.columns]

pc_tcga  = pd.read_csv(TCGA_PC_FILE)
pc_cptac = pd.read_csv(CPTAC_PC_FILE)
pc_tcga['pid']  = pc_tcga['patient_id'].astype(str).str[:12]
pc_cptac['pid'] = pc_cptac['patient_id'].astype(str)

for df in [pc_tcga, pc_cptac]:
    df['PC6_z'] = (df['AE_PC6'] - df['AE_PC6'].mean()) / df['AE_PC6'].std()

def align_pc6_rna(pc_df, rna_df):
    common  = list(set(pc_df['pid']) & set(rna_df.columns))
    pc_sub  = pc_df.set_index('pid').loc[common, 'PC6_z']
    rna_sub = rna_df[common].T.astype(float)
    rna_sub = np.log2(rna_sub + 1)
    return pc_sub, rna_sub

pc6_t, rna_t = align_pc6_rna(pc_tcga,  rna_tcga)
pc6_c, rna_c = align_pc6_rna(pc_cptac, rna_cptac)

print(f"  TCGA: n = {len(pc6_t)}")
print(f"  CPTAC-3: n = {len(pc6_c)}")

# ╔══════════════════════════════════════════════════════════════╗
# ║          Gene-level Spearman → Pathway summary  (Panel A)     ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_gene_corr(pc6, rna):
    all_genes = list({g for genes in GENE_SETS.values() for g in genes})
    results = {}
    for gene in all_genes:
        if gene not in rna.columns: continue
        x = pc6.values.astype(float)
        y = rna[gene].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10: continue
        r, p = spearmanr(x[mask], y[mask])
        results[gene] = {'r': r, 'p': p}
    df = pd.DataFrame(results).T
    _, q, _, _ = multipletests(df['p'].values, method='fdr_bh')
    df['q'] = q
    return df

print("Computing gene-level Spearman correlations...")
corr_t = compute_gene_corr(pc6_t, rna_t)
corr_c = compute_gene_corr(pc6_c, rna_c)

def pathway_summary(corr_df):
    rows = []
    for name, genes in GENE_SETS.items():
        avail = [g for g in genes if g in corr_df.index]
        if not avail: continue
        rows.append({'pathway': name,
                     'mean_r': corr_df.loc[avail, 'r'].mean(),
                     'n': len(avail)})
    return pd.DataFrame(rows)

# ★ 두 cohort 모두 동일한 함수로 계산 — 하드코딩 없음
summ_t = pathway_summary(corr_t)
summ_c = pathway_summary(corr_c)

# ╔══════════════════════════════════════════════════════════════╗
# ║              ssGSEA NES ~ PC6  (Panel B)                      ║
# ╚══════════════════════════════════════════════════════════════╝

print("\nRunning ssGSEA...")
ssgsea_rows    = []
ssgsea_results = {}
RUN_SSGSEA     = False

try:
    import gseapy as gp

    gene_set_upper = {k: [g.upper() for g in v] for k, v in GENE_SETS.items()}

    # ★ 두 cohort 모두 동일한 ssGSEA pipeline — 하드코딩 없음
    for coh_name, rna_df, pc6 in [('TCGA', rna_t, pc6_t),
                                    ('CPTAC-3', rna_c, pc6_c)]:
        expr = rna_df.T.copy()
        expr.index = expr.index.str.upper()

        ss = gp.ssgsea(
            data=expr,
            gene_sets=gene_set_upper,
            outdir=None,
            sample_norm_method='rank',
            no_plot=True,
            min_size=2,
            verbose=False
        )

        scores = ss.res2d.pivot(index='Term', columns='Name', values='NES').T.astype(float)
        common  = list(set(scores.index) & set(pc6.index))
        sc_sub  = scores.loc[common]
        pc6_sub = pc6.loc[common].astype(float)

        coh_pvals = []; coh_data = []
        for pathway in GENE_SETS.keys():
            if pathway not in sc_sub.columns: continue
            x = pc6_sub.values.astype(float)
            y = sc_sub[pathway].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5: continue
            r, p = spearmanr(x[mask], y[mask])
            coh_pvals.append(p)
            coh_data.append({'cohort': coh_name, 'pathway': pathway,
                              'r': round(r, 3), 'p': round(p, 4),
                              'n': int(mask.sum())})

        if coh_pvals:
            _, qvals, _, _ = multipletests(coh_pvals, method='fdr_bh')
            for i, row in enumerate(coh_data):
                row['q'] = round(qvals[i], 4)
                row['sig'] = ('***' if row['q']<0.001 else '**' if row['q']<0.01
                               else '*' if row['q']<0.05 else 'ns')
                ssgsea_rows.append(row)

        ssgsea_results[coh_name] = (pd.DataFrame(coh_data).set_index('pathway')
                                     if coh_data else pd.DataFrame())

    pd.DataFrame(ssgsea_rows).to_csv(OUT_SSGSEA, index=False)
    RUN_SSGSEA = True
    print("  ssGSEA complete!")

except ImportError:
    print("  gseapy 미설치. pip install gseapy 후 재실행.")
except Exception as e:
    print(f"  ssGSEA 오류: {e}")

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 레이아웃 (1행 × 2열)                 ║
# ╚══════════════════════════════════════════════════════════════╝

fig = plt.figure(figsize=(22, 9))
fig.patch.set_facecolor('white')

gs = GridSpec(1, 2, figure=fig,
              wspace=0.30,
              left=0.07, right=0.97,
              top=0.84, bottom=0.13)

LS, TS, TIS, PS = 20, 18, 21, 24
TITLE_PAD = 14

def add_panel_label(ax, letter):
    ax.set_title(letter, loc='left', fontsize=PS, fontweight='bold', pad=TITLE_PAD)

# ════════════════════════════════════════════════════════════════
# ── A: Pathway-level mean r ────────────────────────────────────
# ════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])
pathways = summ_t['pathway'].tolist()
x = np.arange(len(pathways)); w = 0.35

b1 = ax_a.bar(x-w/2, summ_t['mean_r'], w,
              color=[SET_COLORS[p] for p in pathways],
              alpha=0.90, edgecolor='white', label='TCGA')
b2 = ax_a.bar(x+w/2, summ_c['mean_r'], w,
              color=[SET_COLORS[p] for p in pathways],
              alpha=0.48, edgecolor='#555', linewidth=0.7,
              hatch='//', label='CPTAC-3')

for bar in list(b1)+list(b2):
    h = bar.get_height()
    ax_a.text(bar.get_x()+bar.get_width()/2,
              h+(0.012 if h>=0 else -0.012),
              f'{h:.2f}', ha='center',
              va='bottom' if h>=0 else 'top',
              fontsize=14, fontweight='bold')

ax_a.axhline(0, color='black', lw=0.9)
ax_a.set_xticks(x)
ax_a.set_xticklabels(pathways, fontsize=16, rotation=15, ha='right')
ax_a.set_ylabel("Mean Spearman r  (vs PC6)", fontsize=LS)
ax_a.set_title('Pathway-level mean r (TCGA vs CPTAC-3)',
               loc='center', fontsize=TIS, fontweight='bold', pad=TITLE_PAD)
add_panel_label(ax_a, 'A')

ax_a.legend(fontsize=15, frameon=True,
            loc='upper right').get_frame().set_edgecolor('#ccc')
ax_a.spines[['top','right']].set_visible(False)
ax_a.tick_params(axis='y', labelsize=TS)

# ════════════════════════════════════════════════════════════════
# ── B: ssGSEA NES ~ PC6 ─────────────────────────────────────────
# ════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1])

if RUN_SSGSEA and ssgsea_results:
    ss_t = ssgsea_results.get('TCGA', pd.DataFrame())
    ss_c = ssgsea_results.get('CPTAC-3', pd.DataFrame())

    pathways_b = list(GENE_SETS.keys())
    x_b = np.arange(len(pathways_b)); w_b = 0.32
    colors_b = [SET_COLORS[p] for p in pathways_b]

    def sig_label(q):
        return '***' if q<0.001 else '**' if q<0.01 else '*' if q<0.05 else 'ns'

    r_t_vals = [ss_t.loc[p,'r'] if p in ss_t.index else 0 for p in pathways_b]
    r_c_vals = [ss_c.loc[p,'r'] if p in ss_c.index else 0 for p in pathways_b]
    q_t_vals = [ss_t.loc[p,'q'] if p in ss_t.index else 1 for p in pathways_b]
    q_c_vals = [ss_c.loc[p,'q'] if p in ss_c.index else 1 for p in pathways_b]

    ax_b.bar(x_b-w_b/2, r_t_vals, w_b,
             color=colors_b, alpha=0.90, edgecolor='white', label='TCGA')
    ax_b.bar(x_b+w_b/2, r_c_vals, w_b,
             color=colors_b, alpha=0.48, edgecolor='#555',
             linewidth=0.7, hatch='//', label='CPTAC-3')

    for i, p in enumerate(pathways_b):
        r_t=r_t_vals[i]; q_t=q_t_vals[i]
        r_c=r_c_vals[i]; q_c=q_c_vals[i]
        ct='#c0392b' if q_t<0.05 else '#888'
        cc='#c0392b' if q_c<0.05 else '#888'
        ax_b.text(x_b[i]-w_b/2, r_t+(0.018 if r_t>=0 else -0.035),
                  sig_label(q_t), ha='center',
                  va='bottom' if r_t>=0 else 'top',
                  fontsize=15, color=ct, fontweight='bold')
        ax_b.text(x_b[i]+w_b/2, r_c+(0.018 if r_c>=0 else -0.035),
                  sig_label(q_c), ha='center',
                  va='bottom' if r_c>=0 else 'top',
                  fontsize=15, color=cc, fontweight='bold')
        if abs(r_t) > 0.07:
            ax_b.text(x_b[i]-w_b/2, r_t/2, f'{r_t:+.2f}',
                      ha='center', va='center', fontsize=13,
                      color='white', fontweight='bold')
        if abs(r_c) > 0.07:
            ax_b.text(x_b[i]+w_b/2, r_c/2, f'{r_c:+.2f}',
                      ha='center', va='center', fontsize=13,
                      color='white', fontweight='bold')

    ax_b.axhline(0, color='black', lw=0.9)
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(pathways_b, fontsize=16, rotation=15, ha='right')
    ax_b.set_ylabel("Spearman r  (ssGSEA NES vs PC6)", fontsize=LS)
    ax_b.set_title('ssGSEA Pathway Enrichment Score ~ PC6',
                   loc='center', fontsize=TIS, fontweight='bold', pad=TITLE_PAD)
    add_panel_label(ax_b, 'B')

    ax_b.legend(fontsize=15, frameon=True,
                loc='upper right').get_frame().set_edgecolor('#ccc')
    ax_b.spines[['top','right']].set_visible(False)
    ax_b.tick_params(axis='y', labelsize=TS)
    ax_b.set_ylim(-0.32, 1.00)

else:
    ax_b.text(0.5, 0.5,
              'ssGSEA not available.\nInstall gseapy: pip install gseapy',
              ha='center', va='center', fontsize=16, color='gray',
              transform=ax_b.transAxes)
    ax_b.set_axis_off()
    add_panel_label(ax_b, 'B')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 제목 & 저장                          ║
# ╚══════════════════════════════════════════════════════════════╝

fig.suptitle(
    'PC6 Represents a Conserved Warburg-like Metabolic Program',
    fontsize=24, fontweight='bold', y=0.96)

plt.savefig(OUT_TIFF, dpi=300, format='tiff',
            bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n저장 완료: {OUT_TIFF}")