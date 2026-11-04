"""
Supplementary Figure. Gene-level Correlation with PC6
======================================================
Panel A: TCGA — gene-level Spearman correlation with PC6
Panel B: CPTAC-3 — same analysis

(원래 Figure 5의 A, B 패널을 supplementary로 분리)

입력:
  RNA_Group1~4_updated_to_new_symbol.csv  (TCGA)
  ex_RNA_Group1~4.csv                     (CPTAC-3)
  ae_latent_pca10_all.csv
  ae_latent_pca10_external.csv

출력:
  FigureS_GeneLevelCorr.tiff
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
OUT_TIFF      = os.path.join(BASE_DIR, "FigureS_GeneLevelCorr.tiff")

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
# ║                    Gene-level Spearman                        ║
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

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 레이아웃                             ║
# ╚══════════════════════════════════════════════════════════════╝

fig = plt.figure(figsize=(20, 11))
fig.patch.set_facecolor('white')

gs = GridSpec(1, 2, figure=fig,
              wspace=0.30,
              left=0.07, right=0.97,
              top=0.83, bottom=0.14)        # ← suptitle/caption 공간 확보

# 글씨 크기 (다른 figure들과 통일)
LS, TS, TIS, PS = 18, 16, 19, 22
TITLE_PAD = 12

def add_panel_label(ax, letter):
    ax.set_title(letter, loc='left', fontsize=PS, fontweight='bold', pad=TITLE_PAD)

def draw_gene_corr(ax, corr_df, title, panel_label, n_samples):
    all_genes, all_r, all_colors = [], [], []
    for name, genes in GENE_SETS.items():
        avail = [g for g in genes if g in corr_df.index]
        for g in avail:
            all_genes.append(g)
            all_r.append(corr_df.loc[g, 'r'])
            all_colors.append(SET_COLORS[name])

    y = np.arange(len(all_genes))
    ax.barh(y, all_r, color=all_colors, alpha=0.82, edgecolor='white', height=0.72)

    pvals = [corr_df.loc[g, 'p'] for g in all_genes]
    _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
    for yi, q in enumerate(qvals):
        mk = '***' if q<0.001 else '**' if q<0.01 else '*' if q<0.05 else ''
        if mk:
            xpos = all_r[yi] + (0.012 if all_r[yi] >= 0 else -0.012)
            ax.text(xpos, yi, mk, fontsize=11, va='center',
                    ha='left' if all_r[yi] >= 0 else 'right',
                    color='#c0392b', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(all_genes, fontsize=12)
    ax.axvline(0, color='black', lw=0.9)
    ax.set_xlabel("Spearman r  (vs PC6)", fontsize=LS)
    ax.set_title(f'{title}\n(n = {n_samples})',
                 loc='center', fontsize=TIS, fontweight='bold', pad=TITLE_PAD)
    add_panel_label(ax, panel_label)

    ax.tick_params(axis='x', labelsize=TS)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(-0.7, 0.8)

    handles = [mpatches.Patch(color=SET_COLORS[n], label=n) for n in GENE_SETS]
    leg = ax.legend(handles=handles, fontsize=12, frameon=True, loc='lower left')
    leg.get_frame().set_edgecolor('#ccc')

# Panel A: TCGA
ax_a = fig.add_subplot(gs[0, 0])
draw_gene_corr(ax_a, corr_t,
               'TCGA: Gene-level correlation with PC6', 'A',
               n_samples=len(pc6_t))

# Panel B: CPTAC-3
ax_b = fig.add_subplot(gs[0, 1])
draw_gene_corr(ax_b, corr_c,
               'CPTAC-3: Gene-level correlation with PC6', 'B',
               n_samples=len(pc6_c))

# ── Caption box (FDR 별표 의미) ────────────────────────────
fig.text(0.5, 0.035,
         "Spearman correlation between PC6 z-score and gene expression (log₂(TPM+1)).  "
         "FDR-adjusted (Benjamini–Hochberg).   ***  q<0.001    **  q<0.01    *  q<0.05",
         ha='center', va='bottom',
         fontsize=12, color='#444', style='italic')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 제목 & 저장                          ║
# ╚══════════════════════════════════════════════════════════════╝

fig.suptitle(
    'Gene-level Correlation between PC6 and Metabolic / Proliferation Programs',
    fontsize=20, fontweight='bold', y=0.92)

plt.savefig(OUT_TIFF, dpi=300, format='tiff',
            bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n저장 완료: {OUT_TIFF}")