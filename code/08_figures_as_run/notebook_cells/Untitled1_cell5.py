"""
Figure 5. PC6 Represents a Conserved Warburg-like Metabolic Program
=====================================================================
Panel A: TCGA — gene-level Spearman correlation with PC6
Panel B: CPTAC-3 — same analysis
Panel C: Pathway-level mean r (TCGA vs CPTAC-3)
Panel D: Warburg vs Proliferation vs OXPHOS comparison (TCGA)
Panel E: ssGSEA NES ~ PC6 Spearman (TCGA vs CPTAC-3)

입력:
  RNA_Group1~4_updated_to_new_symbol.csv  (TCGA)
  ex_RNA_Group1~4.csv                     (CPTAC-3)
  ae_latent_pca10_all.csv
  ae_latent_pca10_external.csv

출력:
  Figure5.tiff

패키지:
  pip install gseapy
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

def pathway_summary(corr_df):
    rows = []
    for name, genes in GENE_SETS.items():
        avail = [g for g in genes if g in corr_df.index]
        if not avail: continue
        rows.append({'pathway': name, 'mean_r': corr_df.loc[avail,'r'].mean(), 'n': len(avail)})
    return pd.DataFrame(rows)

summ_t = pathway_summary(corr_t)
summ_c = pathway_summary(corr_c)

# ╔══════════════════════════════════════════════════════════════╗
# ║          ssGSEA enrichment score ~ PC6  (Panel E)            ║
# ╚══════════════════════════════════════════════════════════════╝

print("\nRunning ssGSEA...")
ssgsea_rows   = []
ssgsea_results = {}
RUN_SSGSEA    = False

try:
    import gseapy as gp

    gene_set_upper = {k: [g.upper() for g in v] for k, v in GENE_SETS.items()}

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
                              'r': round(r, 3), 'p': round(p, 4), 'n': int(mask.sum())})

        if coh_pvals:
            _, qvals, _, _ = multipletests(coh_pvals, method='fdr_bh')
            for i, row in enumerate(coh_data):
                row['q'] = round(qvals[i], 4)
                row['sig'] = ('***' if row['q']<0.001 else '**' if row['q']<0.01
                               else '*' if row['q']<0.05 else 'ns')
                ssgsea_rows.append(row)

        # ★ 핵심: pathway를 index로 설정해서 저장
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
# ║   Figure 레이아웃: 3행 × 2열                                   ║
# ║   Row 0: A, B  (gene-level barplot)                           ║
# ║   Row 1: C, D  (pathway mean r / dot plot)                    ║
# ║   Row 2: E     (ssGSEA, full width)                           ║
# ╚══════════════════════════════════════════════════════════════╝

fig = plt.figure(figsize=(22, 15))
fig.patch.set_facecolor('white')

gs = GridSpec(3, 2, figure=fig,
              hspace=0.48, wspace=0.35,
              left=0.07, right=0.97, top=0.93, bottom=0.06,
              height_ratios=[1.2, 1.0, 0.9])

LS, TS, TIS, PS = 11, 9.0, 11.5, 14

# ── A & B ─────────────────────────────────────────────────
def draw_gene_corr(ax, corr_df, title, panel_label):
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
            xpos = all_r[yi] + (0.01 if all_r[yi] >= 0 else -0.01)
            ax.text(xpos, yi, mk, fontsize=7, va='center',
                    ha='left' if all_r[yi] >= 0 else 'right',
                    color='#333', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(all_genes, fontsize=8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel("Spearman r  (vs PC6)", fontsize=LS)
    ax.set_title(title, fontsize=TIS, fontweight='bold')
    ax.tick_params(axis='x', labelsize=TS)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(-0.6, 0.7)
    handles = [mpatches.Patch(color=SET_COLORS[n], label=n) for n in GENE_SETS]
    leg = ax.legend(handles=handles, fontsize=8, frameon=True, loc='lower right')
    leg.get_frame().set_edgecolor('#ccc')
    ax.text(-0.10, 1.04, panel_label, transform=ax.transAxes,
            fontsize=PS, fontweight='bold')

ax_a = fig.add_subplot(gs[0, 0])
draw_gene_corr(ax_a, corr_t, 'TCGA: Gene-level correlation with PC6', 'A')
ax_b = fig.add_subplot(gs[0, 1])
draw_gene_corr(ax_b, corr_c, 'CPTAC-3: Gene-level correlation with PC6', 'B')

# ── C ─────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
pathways = summ_t['pathway'].tolist()
x = np.arange(len(pathways)); w = 0.35
b1 = ax_c.bar(x-w/2, summ_t['mean_r'], w,
              color=[SET_COLORS[p] for p in pathways],
              alpha=0.90, edgecolor='white', label='TCGA')
b2 = ax_c.bar(x+w/2, summ_c['mean_r'], w,
              color=[SET_COLORS[p] for p in pathways],
              alpha=0.48, edgecolor='#555', linewidth=0.7,
              hatch='//', label='CPTAC-3')
for bar in list(b1)+list(b2):
    h = bar.get_height()
    ax_c.text(bar.get_x()+bar.get_width()/2,
              h+(0.005 if h>=0 else -0.005),
              f'{h:.2f}', ha='center',
              va='bottom' if h>=0 else 'top',
              fontsize=8, fontweight='bold')
ax_c.axhline(0, color='black', lw=0.8)
ax_c.set_xticks(x)
ax_c.set_xticklabels(pathways, fontsize=9.5, rotation=15, ha='right')
ax_c.set_ylabel("Mean Spearman r  (vs PC6)", fontsize=LS)
ax_c.set_title('Pathway-level mean r (TCGA vs CPTAC-3)', fontsize=TIS, fontweight='bold')
ax_c.legend(fontsize=9.5, frameon=True)
ax_c.spines[['top','right']].set_visible(False)
ax_c.tick_params(axis='y', labelsize=TS)
ax_c.text(-0.10, 1.04, 'C', transform=ax_c.transAxes, fontsize=PS, fontweight='bold')

# ── D ─────────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
warburg_genes = (GENE_SETS['Glycolysis'] +
                 GENE_SETS['Hypoxia'] +
                 GENE_SETS['Lactate/MCT'])
np.random.seed(42)
for yi, (genes, col) in enumerate([(warburg_genes,            '#E74C3C'),
                                    (GENE_SETS['Proliferation'],'#7F8C8D'),
                                    (GENE_SETS['OXPHOS'],       '#2980B9')]):
    for gene in genes:
        if gene not in corr_t.index: continue
        r = corr_t.loc[gene, 'r']
        ax_d.scatter(r, yi+np.random.uniform(-0.15,0.15),
                     color=col, s=55, alpha=0.75,
                     edgecolors='white', linewidths=0.4, zorder=3)
        ax_d.text(r, yi+0.18, gene, fontsize=7, ha='center',
                  va='bottom', color=col, rotation=45)
    rs = [corr_t.loc[g,'r'] for g in genes if g in corr_t.index]
    if rs:
        ax_d.plot([np.mean(rs)],[yi], marker='|', color=col,
                  markersize=20, markeredgewidth=3, zorder=5)
        ax_d.text(np.mean(rs), yi-0.25,
                  f'mean r = {np.mean(rs):.2f}',
                  ha='center', va='top', fontsize=8.5,
                  color=col, fontweight='bold')
ax_d.axvline(0, color='black', lw=0.8, ls='--', alpha=0.6)
ax_d.set_yticks([0,1,2])
ax_d.set_yticklabels(['Warburg\n(Glycolysis + Hypoxia + MCT)',
                       'Proliferation\n(E2F / G2M / MKI67)',
                       'OXPHOS'], fontsize=9.5)
ax_d.set_xlabel("Spearman r  (vs PC6, TCGA)", fontsize=LS)
ax_d.set_title('PC6 ~ Warburg vs Proliferation vs OXPHOS\n'
               '(TCGA, proliferation as negative control)',
               fontsize=TIS, fontweight='bold')
ax_d.set_xlim(-0.7, 0.75); ax_d.set_ylim(-0.55, 2.7)
ax_d.spines[['top','right']].set_visible(False)
ax_d.tick_params(axis='x', labelsize=TS)
ax_d.text(-0.10, 1.04, 'D', transform=ax_d.transAxes, fontsize=PS, fontweight='bold')

# ── E  ssGSEA (full width) ────────────────────────────────
ax_e = fig.add_subplot(gs[2, :])

if RUN_SSGSEA and ssgsea_results:
    ss_t = ssgsea_results.get('TCGA', pd.DataFrame())
    ss_c = ssgsea_results.get('CPTAC-3', pd.DataFrame())

    pathways_e = list(GENE_SETS.keys())
    x_e = np.arange(len(pathways_e)); w_e = 0.32
    colors_e = [SET_COLORS[p] for p in pathways_e]

    def sig_label(q):
        return '***' if q<0.001 else '**' if q<0.01 else '*' if q<0.05 else 'ns'

    r_t_vals = [ss_t.loc[p,'r'] if p in ss_t.index else 0 for p in pathways_e]
    r_c_vals = [ss_c.loc[p,'r'] if p in ss_c.index else 0 for p in pathways_e]
    q_t_vals = [ss_t.loc[p,'q'] if p in ss_t.index else 1 for p in pathways_e]
    q_c_vals = [ss_c.loc[p,'q'] if p in ss_c.index else 1 for p in pathways_e]

    b3 = ax_e.bar(x_e-w_e/2, r_t_vals, w_e,
                  color=colors_e, alpha=0.90, edgecolor='white', label='TCGA')
    b4 = ax_e.bar(x_e+w_e/2, r_c_vals, w_e,
                  color=colors_e, alpha=0.48, edgecolor='#555',
                  linewidth=0.7, hatch='//', label='CPTAC-3')

    for i, p in enumerate(pathways_e):
        r_t=r_t_vals[i]; q_t=q_t_vals[i]
        r_c=r_c_vals[i]; q_c=q_c_vals[i]
        ct='#c0392b' if q_t<0.05 else '#888'
        cc='#c0392b' if q_c<0.05 else '#888'
        ax_e.text(x_e[i]-w_e/2, r_t+(0.012 if r_t>=0 else -0.025),
                  sig_label(q_t), ha='center',
                  va='bottom' if r_t>=0 else 'top',
                  fontsize=11, color=ct, fontweight='bold')
        ax_e.text(x_e[i]+w_e/2, r_c+(0.012 if r_c>=0 else -0.025),
                  sig_label(q_c), ha='center',
                  va='bottom' if r_c>=0 else 'top',
                  fontsize=11, color=cc, fontweight='bold')
        if abs(r_t) > 0.06:
            ax_e.text(x_e[i]-w_e/2, r_t/2, f'{r_t:+.2f}',
                      ha='center', va='center', fontsize=9,
                      color='white', fontweight='bold')
        if abs(r_c) > 0.06:
            ax_e.text(x_e[i]+w_e/2, r_c/2, f'{r_c:+.2f}',
                      ha='center', va='center', fontsize=9,
                      color='white', fontweight='bold')

    ax_e.axhline(0, color='black', lw=0.8)
    ax_e.set_xticks(x_e)
    ax_e.set_xticklabels(pathways_e, fontsize=12)
    ax_e.set_ylabel("Spearman r  (ssGSEA NES vs PC6)", fontsize=LS)
    ax_e.set_title(
        'ssGSEA Pathway Enrichment Score ~ PC6  (Panel E)\n'
        'Single-sample GSEA, rank normalization | FDR-corrected (Benjamini-Hochberg)',
        fontsize=TIS, fontweight='bold')
    ax_e.legend(fontsize=10, frameon=True,
                loc='upper right').get_frame().set_edgecolor('#ccc')
    ax_e.spines[['top','right']].set_visible(False)
    ax_e.tick_params(axis='y', labelsize=TS)
    ax_e.set_ylim(-0.25, 0.88)

else:
    ax_e.text(0.5, 0.5,
              'ssGSEA not available.\nInstall gseapy: pip install gseapy',
              ha='center', va='center', fontsize=14, color='gray',
              transform=ax_e.transAxes)
    ax_e.set_axis_off()

ax_e.text(-0.04, 1.04, 'E', transform=ax_e.transAxes,
          fontsize=PS, fontweight='bold')

# ╔══════════════════════════════════════════════════════════════╗
# ║            제목만, 부연설명 없음                                ║
# ╚══════════════════════════════════════════════════════════════╝

fig.suptitle(
    'PC6 Represents a Conserved Warburg-like Metabolic Program',
    fontsize=12.5, fontweight='bold', y=0.99)

plt.savefig(OUT_TIFF, dpi=300, format='tiff',
            bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n저장 완료: {OUT_TIFF}")