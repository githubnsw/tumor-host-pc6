"""
Figure 6. Joint Effects of PC6 and Body Composition  (Standalone)
==================================================================
사전 계산 CSV 없이 전부 인라인으로 계산합니다.

Panel A: PC6 × SMA 4-group KM survival
Panel B: Volcano plot (PC6-High/SMA-Low vs Others)
Panel C: GSEA NES barplot (Hallmark + Immune gene sets)
Panel D: Transcriptional signature heatmap (4 groups)
Panel E: Representative marker violin (4 groups)

입력 (BASE_DIR 안에 있어야 함):
  TCGA_with_genes.csv
  ae_latent_pca10_all.csv
  RNA_Group1~4_updated_to_new_symbol.csv  (4개)

출력:
  Figure6.tiff  (dpi=300)

패키지:
  pip install lifelines scipy statsmodels
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import kruskal, ranksums
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    파일 경로 설정                              ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_DIR = r"C:\Users\nsw\Desktop\body_composition"

TCGA_FILE    = os.path.join(BASE_DIR, "TCGA_with_genes.csv")
PC_TCGA_FILE = os.path.join(BASE_DIR, "ae_latent_pca10_all.csv")
TCGA_RNA_FILES = [
    os.path.join(BASE_DIR, f"RNA_Group{i}_updated_to_new_symbol.csv")
    for i in range(1, 5)
]
OUT_TIFF = os.path.join(BASE_DIR, "Figure6.tiff")

# ╔══════════════════════════════════════════════════════════════╗
# ║                    데이터 로드                                 ║
# ╚══════════════════════════════════════════════════════════════╝

print("Loading data...")
dfs = [pd.read_csv(f).set_index('gene') for f in TCGA_RNA_FILES]
rna = pd.concat(dfs, axis=1).loc[lambda x: ~x.index.duplicated()]
rna.columns = [c[:12] for c in rna.columns]
rna_log = np.log2(rna + 1)

pc   = pd.read_csv(PC_TCGA_FILE)
clin = pd.read_csv(TCGA_FILE)
pc['pid']   = pc['patient_id'].astype(str).str[:12]
clin['pid'] = clin['patient_id'].astype(str).str[:12]
pc['PC6_z'] = (pc['AE_PC6'] - pc['AE_PC6'].mean()) / pc['AE_PC6'].std()

merged = clin[['pid','OS_time','OS','SMA_zscore','cancer_type']].merge(
    pc[['pid','PC6_z']], on='pid')
merged['OS_months'] = merged['OS_time'] / 30.44
merged = merged.dropna(subset=['PC6_z','SMA_zscore','OS_months','OS'])

pc6_med = merged['PC6_z'].median()
sma_med = merged['SMA_zscore'].median()
merged['PC6_grp'] = np.where(merged['PC6_z'] >= pc6_med, 'PC6-High', 'PC6-Low')
merged['SMA_grp'] = np.where(merged['SMA_zscore'] >= sma_med, 'SMA-High', 'SMA-Low')
merged['group4']  = merged['PC6_grp'] + ' / ' + merged['SMA_grp']

G_ORDER  = ['PC6-Low / SMA-High','PC6-Low / SMA-Low',
            'PC6-High / SMA-High','PC6-High / SMA-Low']
G_LABELS = ['PC6-Low\nSMA-High','PC6-Low\nSMA-Low',
            'PC6-High\nSMA-High','PC6-High\nSMA-Low']
G_COLORS = {'PC6-Low / SMA-High':'#2166AC','PC6-Low / SMA-Low':'#74ADD1',
            'PC6-High / SMA-High':'#F46D43','PC6-High / SMA-Low':'#D73027'}

# ╔══════════════════════════════════════════════════════════════╗
# ║               DEG 계산 (PC6-High/SMA-Low vs Others)           ║
# ╚══════════════════════════════════════════════════════════════╝

print("Computing DEG...")
worst_pids = set(merged[merged['group4']=='PC6-High / SMA-Low']['pid'])
other_pids = set(merged[merged['group4']!='PC6-High / SMA-Low']['pid'])
worst_cols = [c for c in rna_log.columns if c in worst_pids]
other_cols = [c for c in rna_log.columns if c in other_pids]

deg_rows = []
for gene in rna_log.index:
    v1 = rna_log.loc[gene, worst_cols].dropna().values.astype(float)
    v2 = rna_log.loc[gene, other_cols].dropna().values.astype(float)
    if len(v1) < 5 or len(v2) < 5:
        continue
    fc  = v1.mean() - v2.mean()
    _, p = ranksums(v1, v2)
    deg_rows.append({'gene': gene, 'logFC': fc, 'p': p})

deg = pd.DataFrame(deg_rows)
_, deg['q'], _, _ = multipletests(deg['p'].values, method='fdr_bh')
deg['sig'] = (deg['q'] < 0.05) & (np.abs(deg['logFC']) >= 0.5)
print(f"  DEG: {deg['sig'].sum()} significant genes")

# ╔══════════════════════════════════════════════════════════════╗
# ║               GSEA 계산 (Hallmark + Immune gene sets)         ║
# ╚══════════════════════════════════════════════════════════════╝

print("Running GSEA...")

HALLMARK_SETS = {
    'HALLMARK_HYPOXIA'                          : ['CA9','NDRG1','VEGFA','HIF1A','ADM','EGLN3','P4HA2','PLOD2','SLC2A1','DDIT4','BNIP3','HMOX1','LDHA','ENO1','PGK1','TGFB1'],
    'HALLMARK_GLYCOLYSIS'                       : ['HK2','PFKP','ALDOA','ENO1','PGK1','PKM','LDHA','GAPDH','TPI1','PGAM1','GPI','ENO2','PFKFB4','LDHB'],
    'HALLMARK_APOPTOSIS'                        : ['CASP3','CASP7','CASP8','CASP9','BCL2','BAX','CYCS','APAF1','TP53'],
    'HALLMARK_INTERFERON_GAMMA_RESPONSE'        : ['STAT1','IRF1','CXCL9','CXCL10','IDO1','CD274','TAP1','B2M'],
    'HALLMARK_INFLAMMATORY_RESPONSE'            : ['IL6','IL1B','TNF','CXCL8','CCL2','PTGS2'],
    'HALLMARK_TNFA_SIGNALING_VIA_NFKB'         : ['NFKB1','RELA','TNF','IL6','CCL2','ICAM1','VCAM1','CXCL1'],
    'HALLMARK_MYC_TARGETS_V1'                  : ['MYC','CDK4','CCNA2','TOP2A','E2F1','NPM1','PCNA'],
    'HALLMARK_ANGIOGENESIS'                     : ['VEGFA','VEGFB','VEGFC','FGF2','ANGPT1','PDGFB'],
    'HALLMARK_E2F_TARGETS'                      : ['E2F1','E2F2','CCNA2','CDK1','MCM2','PCNA','TOP2A'],
    'HALLMARK_TGF_BETA_SIGNALING'               : ['TGFB1','TGFB2','SMAD2','SMAD3','SMAD4'],
    'HALLMARK_OXIDATIVE_PHOSPHORYLATION'        : ['SDHA','SDHB','COX5A','NDUFB3','UQCRC1'],
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION': ['VIM','FN1','SNAI1','TWIST1','ZEB1','CDH2','MMP2'],
    'HALLMARK_WNT_BETA_CATENIN_SIGNALING'       : ['CTNNB1','APC','AXIN1','LRP5','FZD1','TCF7'],
}
IMMUNE_SETS = {
    'CD8_T_CELL'               : ['CD8A','CD8B','GZMA','GZMB','PRF1','IFNG'],
    'TREG'                     : ['FOXP3','IL2RA','IKZF2','CTLA4'],
    'T_CELL_EXHAUSTION'        : ['HAVCR2','PDCD1','LAG3','TOX','TIGIT'],
    'M2_MACROPHAGE'            : ['MSR1','CD163','MRC1','IL10'],
    'M1_MACROPHAGE'            : ['TNF','IL6','IL12A','NOS2','CD80'],
    'MYELOID_IMMUNOSUPPRESSION': ['ARG1','IL10','TGFB1','IDO1'],
    'NK_CELL'                  : ['NCAM1','KLRB1','KLRD1','NKG7','GNLY'],
    'INFLAMMATORY_CYTOKINE'    : ['IL6','IL1B','TNF','CCL2','CXCL8','IL18'],
    'CACHEXIA_SIGNAL'          : ['MSTN','FBXO32','IL6','ACVR2B'],
}

def run_gsea_simple(deg_df, gene_sets):
    scores = deg_df.set_index('gene')['logFC'].to_dict()
    results = []
    for term, genes in gene_sets.items():
        avail = [g for g in genes if g in scores]
        if len(avail) < 3:
            continue
        in_s  = [scores[g] for g in avail]
        out_s = [scores[g] for g in scores if g not in set(genes)]
        if len(out_s) < 10:
            continue
        stat, p = ranksums(in_s, out_s)
        results.append({'Term': term, 'NES': stat, 'FDR q-val': p})
    df = pd.DataFrame(results)
    if len(df) > 0:
        _, df['FDR q-val'], _, _ = multipletests(df['FDR q-val'].values, method='fdr_bh')
    return df

res_h = run_gsea_simple(deg, HALLMARK_SETS)
res_i = run_gsea_simple(deg, IMMUNE_SETS)

# ╔══════════════════════════════════════════════════════════════╗
# ║               Heatmap z-score 계산                            ║
# ╚══════════════════════════════════════════════════════════════╝

print("Computing heatmap z-scores...")

SIGS = {
    'Glycolysis'       : 'HK2,PKM,ALDOA,LDHA,GAPDH,ENO1'.split(','),
    'Hypoxia'          : 'CA9,NDRG1,VEGFA,HIF1A,ADM'.split(','),
    'Lactate/MCT'      : 'LDHB,SLC16A3,SLC16A1'.split(','),
    'CD8 T cell'       : 'CD8A,CD8B,GZMA,GZMB,PRF1'.split(','),
    'T cell Exhaustion': 'HAVCR2,PDCD1,LAG3,TOX'.split(','),
    'Treg'             : 'FOXP3,IL2RA,IKZF2'.split(','),
    'M2 Macrophage'    : 'MSR1,CD163,MRC1'.split(','),
    'IFN-γ Response'   : 'STAT1,IRF1,CXCL9,CXCL10,IDO1'.split(','),
    'Cachexia Signal'  : 'MSTN,FBXO32,IL6,ACVR2B'.split(','),
    'TGF-β'            : 'TGFB1,TGFB2,SMAD2,SMAD3'.split(','),
}

hm_rows = {}
for sig_name, genes in SIGS.items():
    avail = [g for g in genes if g in rna_log.index]
    if not avail:
        continue
    row, all_vals = {}, []
    for grp in G_ORDER:
        pids = [p for p in merged[merged['group4']==grp]['pid'] if p in rna_log.columns]
        val  = rna_log.loc[avail, pids].values.mean()
        row[grp] = val
        all_vals.append(val)
    mu, sd = np.mean(all_vals), np.std(all_vals)
    for grp in G_ORDER:
        row[grp] = (row[grp] - mu) / (sd + 1e-9)
    hm_rows[sig_name] = row

heatmap_z = pd.DataFrame(hm_rows).T   # sigs × groups

SHORT_NAME = {
    'HALLMARK_HYPOXIA':'Hypoxia','HALLMARK_GLYCOLYSIS':'Glycolysis',
    'HALLMARK_APOPTOSIS':'Apoptosis',
    'HALLMARK_INTERFERON_GAMMA_RESPONSE':'IFN-γ Response',
    'HALLMARK_INFLAMMATORY_RESPONSE':'Inflammatory',
    'HALLMARK_TNFA_SIGNALING_VIA_NFKB':'TNFα/NF-κB',
    'HALLMARK_MYC_TARGETS_V1':'MYC Targets',
    'HALLMARK_ANGIOGENESIS':'Angiogenesis',
    'HALLMARK_E2F_TARGETS':'E2F Targets',
    'HALLMARK_TGF_BETA_SIGNALING':'TGF-β',
    'HALLMARK_OXIDATIVE_PHOSPHORYLATION':'OXPHOS',
    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION':'EMT',
    'HALLMARK_WNT_BETA_CATENIN_SIGNALING':'WNT/β-catenin',
    'CD8_T_CELL':'CD8 T Cell','TREG':'Treg',
    'T_CELL_EXHAUSTION':'T Cell Exhaustion',
    'M2_MACROPHAGE':'M2 Macrophage','M1_MACROPHAGE':'M1 Macrophage',
    'MYELOID_IMMUNOSUPPRESSION':'Myeloid Suppress.',
    'NK_CELL':'NK Cell','INFLAMMATORY_CYTOKINE':'Inflam. Cytokine',
    'CACHEXIA_SIGNAL':'Cachexia Signal',
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 레이아웃                             ║
# ╚══════════════════════════════════════════════════════════════╝

print("Drawing figure...")
fig = plt.figure(figsize=(24, 16))
fig.patch.set_facecolor('white')
gs = GridSpec(2, 6, figure=fig,
              hspace=0.30, wspace=0.38,
              left=0.06, right=0.97, top=0.93, bottom=0.05,
              height_ratios=[1.05, 1.0])
LS, TS, TIS, PS = 11, 9, 11.5, 14

# ── A: KM ─────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0:2])
kmf = KaplanMeierFitter()
for grp in G_ORDER:
    sub = merged[merged['group4']==grp]
    kmf.fit(sub['OS_months'], sub['OS'], label=f"{grp} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax_a, color=G_COLORS[grp], lw=2.2,
                               ci_show=True, ci_alpha=0.10,
                               show_censors=True, censor_styles={'ms':4,'marker':'+'})
lrp = multivariate_logrank_test(merged['OS_months'], merged['group4'], merged['OS']).p_value
ax_a.text(0.97, 0.97, f'Log-rank p={lrp:.2e}',
          transform=ax_a.transAxes, fontsize=8.5, va='top', ha='right',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))
ax_a.set_xlabel("Time (months)", fontsize=LS)
ax_a.set_ylabel("Overall Survival Probability", fontsize=LS)
ax_a.set_title('PC6 × SMA 4-Group Survival\n(worst = PC6-High / SMA-Low)', fontsize=TIS, fontweight='bold')
ax_a.set_ylim(-0.08, 1.05)
ax_a.spines[['top','right']].set_visible(False)
ax_a.legend(fontsize=7, loc='upper right').get_frame().set_edgecolor('#ccc')
ax_a.text(-0.10, 1.04, 'A', transform=ax_a.transAxes, fontsize=PS, fontweight='bold')

# ── B: Volcano ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2:4])
up = (deg['q']<0.05) & (deg['logFC']>0.5)
dn = (deg['q']<0.05) & (deg['logFC']<-0.5)
ns_mask = ~(up|dn)
ax_b.scatter(deg.loc[ns_mask,'logFC'], -np.log10(deg.loc[ns_mask,'p'].clip(1e-300)),
             c='#CCCCCC', s=3, alpha=0.4, rasterized=True)
ax_b.scatter(deg.loc[up,'logFC'], -np.log10(deg.loc[up,'p'].clip(1e-300)),
             c='#D73027', s=8, alpha=0.75, rasterized=True, label=f'Up ({up.sum()})')
ax_b.scatter(deg.loc[dn,'logFC'], -np.log10(deg.loc[dn,'p'].clip(1e-300)),
             c='#2166AC', s=8, alpha=0.75, rasterized=True, label=f'Down ({dn.sum()})')
for g, col in [('CA9','#D73027'),('NDRG1','#D73027'),('SLC16A3','#D73027'),
               ('VEGFA','#D73027'),('ALDOA','#D73027'),('PKM','#D73027'),
               ('HAVCR2','#D73027'),('MSTN','#D73027'),('MRC1','#2166AC')]:
    row = deg[deg['gene']==g]
    if len(row)==0: continue
    r = row.iloc[0]
    ax_b.annotate(g, (r['logFC'], -np.log10(r['p']+1e-300)),
                  xytext=(4,2), textcoords='offset points',
                  fontsize=7.5, fontweight='bold', color=col)
foxp3 = deg[deg['gene']=='FOXP3']
if len(foxp3) > 0:
    fx = foxp3.iloc[0]['logFC']; fy = -np.log10(foxp3.iloc[0]['p']+1e-300)
    ax_b.scatter(fx, fy, c='#2166AC', s=60, zorder=5, edgecolors='white', linewidths=1.2)
    ax_b.annotate('FOXP3', xy=(fx, fy), xytext=(fx-0.55, fy+1.2),
                  fontsize=9, fontweight='bold', color='#2166AC',
                  bbox=dict(boxstyle='round,pad=0.25', fc='#EBF5FB', ec='#2166AC', alpha=0.92),
                  arrowprops=dict(arrowstyle='->', color='#2166AC', lw=1.5))
ax_b.axvline(0.5, color='gray', lw=0.7, ls='--', alpha=0.6)
ax_b.axvline(-0.5, color='gray', lw=0.7, ls='--', alpha=0.6)
ax_b.axhline(-np.log10(0.05), color='gray', lw=0.7, ls='--', alpha=0.6)
ax_b.set_xlabel("log₂FC (PC6-High/SMA-Low vs Others)", fontsize=LS)
ax_b.set_ylabel("-log₁₀(p)", fontsize=LS)
ax_b.set_title('DEG: PC6-High/SMA-Low vs All Others\n(FDR<0.05, |log₂FC|≥0.5)', fontsize=TIS, fontweight='bold')
ax_b.legend(fontsize=9, markerscale=2).get_frame().set_edgecolor('#ccc')
ax_b.spines[['top','right']].set_visible(False)
ax_b.text(-0.10, 1.04, 'B', transform=ax_b.transAxes, fontsize=PS, fontweight='bold')

# ── C: GSEA ───────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 4:6])
gsea_comb = pd.concat([res_h, res_i], ignore_index=True).sort_values('NES')
gsea_comb['label'] = gsea_comb['Term'].map(SHORT_NAME).fillna(gsea_comb['Term'])
cols_c = ['#D73027' if (n>0 and q<0.05) else '#F4A582' if n>0
          else '#2166AC' if q<0.05 else '#92C5DE'
          for n,q in zip(gsea_comb['NES'], gsea_comb['FDR q-val'])]
ax_c.barh(range(len(gsea_comb)), gsea_comb['NES'], color=cols_c, alpha=0.88, edgecolor='white')
ax_c.set_yticks(range(len(gsea_comb)))
ax_c.set_yticklabels(gsea_comb['label'], fontsize=8.5)
ax_c.axvline(0, color='black', lw=0.8)
for i, (_, row) in enumerate(gsea_comb.iterrows()):
    q = row['FDR q-val']
    sig = '***' if q<0.001 else '**' if q<0.01 else '*' if q<0.05 else ''
    if sig:
        x = row['NES'] + (0.04 if row['NES']>0 else -0.04)
        ax_c.text(x, i, sig, va='center', ha='left' if row['NES']>0 else 'right',
                  fontsize=9, color='#333', fontweight='bold')
ax_c.set_xlabel("NES", fontsize=LS)
ax_c.set_title('Pathway Enrichment (GSEA)\nPC6-High/SMA-Low vs Others', fontsize=TIS, fontweight='bold')
ax_c.spines[['top','right']].set_visible(False)
sig_p = mpatches.Patch(color='#D73027', label='Enriched FDR<0.05')
ns_p  = mpatches.Patch(color='#F4A582', label='Enriched ns')
dep_p = mpatches.Patch(color='#2166AC', label='Depleted FDR<0.05')
ax_c.legend(handles=[sig_p,ns_p,dep_p], fontsize=7.5, loc='lower right').get_frame().set_edgecolor('#ccc')
ax_c.text(-0.10, 1.04, 'C', transform=ax_c.transAxes, fontsize=PS, fontweight='bold')

# ── D: Heatmap ────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1:3])
hm = heatmap_z[G_ORDER]
im = ax_d.imshow(hm.values, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax_d.set_xticks(range(4))
ax_d.set_xticklabels(G_LABELS, fontsize=8.5)
ax_d.set_yticks(range(len(hm)))
ax_d.set_yticklabels(hm.index, fontsize=9)
for i, grp in enumerate(G_ORDER):
    ax_d.get_xticklabels()[i].set_color(G_COLORS[grp])
    ax_d.get_xticklabels()[i].set_fontweight('bold')
plt.colorbar(im, ax=ax_d, shrink=0.75, pad=0.02, label='z-score')
ax_d.set_title('Transcriptional Signature Heatmap\n(z-score across 4 groups)', fontsize=TIS, fontweight='bold')
for si, sig_name in enumerate(hm.index):
    genes = SIGS.get(sig_name, [])
    avail = [g for g in genes if g in rna_log.index]
    if not avail: continue
    gv = [rna_log.loc[avail,
          [p for p in merged[merged['group4']==g]['pid'] if p in rna_log.columns]
          ].values.flatten() for g in G_ORDER]
    try:
        _, kp = kruskal(*gv)
        sig = '***' if kp<0.001 else '**' if kp<0.01 else '*' if kp<0.05 else ''
        if sig:
            ax_d.text(3.65, si, sig, va='center', ha='left', fontsize=8, color='#333', fontweight='bold')
    except:
        pass
ax_d.text(-0.15, 1.04, 'D', transform=ax_d.transAxes, fontsize=PS, fontweight='bold')

# ── E: Violin ─────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 3:5])
markers = [('LDHA','Glycolysis','#E74C3C'),
           ('SLC16A3','MCT4','#C0392B'),
           ('HAVCR2','T Exhaustion','#8E44AD'),
           ('FOXP3','Treg','#27AE60'),
           ('MSTN','Cachexia','#7F8C8D')]
group_x_base = np.arange(len(markers)) * 5.5
for mi, (gene, cat, col) in enumerate(markers):
    if gene not in rna_log.index: continue
    for gi, grp in enumerate(G_ORDER):
        pids = [p for p in merged[merged['group4']==grp]['pid'] if p in rna_log.columns]
        vals = rna_log.loc[gene, pids].values
        xpos = group_x_base[mi] + gi
        vp = ax_e.violinplot([vals], positions=[xpos], widths=0.8,
                              showmedians=True, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(G_COLORS[grp])
            body.set_alpha(0.80 if grp=='PC6-High / SMA-Low' else 0.42)
            body.set_edgecolor('white')
        vp['cmedians'].set_color(G_COLORS[grp]); vp['cmedians'].set_linewidth(2)
    groups_vals = [rna_log.loc[gene,
                   [p for p in merged[merged['group4']==g]['pid'] if p in rna_log.columns]
                   ].values for g in G_ORDER]
    _, kp = kruskal(*groups_vals)
    sig = '***' if kp<0.001 else '**' if kp<0.01 else '*' if kp<0.05 else 'ns'
    ymax = max(v.max() for v in groups_vals)
    ax_e.text(group_x_base[mi]+1.5, ymax+0.15, sig,
              ha='center', fontsize=11, fontweight='bold',
              color='#c0392b' if sig!='ns' else '#888')
    ax_e.text(group_x_base[mi]+1.5, -0.7,
              f'{gene}\n({cat})', ha='center', fontsize=8.5, fontweight='bold', color=col)
ax_e.set_ylabel("log₂(TPM+1)", fontsize=LS)
ax_e.set_title('Representative Marker Expression\nby PC6 × SMA 4 Groups', fontsize=TIS, fontweight='bold')
ax_e.set_xlim(-0.8, group_x_base[-1]+3.8)
ax_e.set_xticks([])
ax_e.spines[['top','right']].set_visible(False)
handles_e = [mpatches.Patch(color=G_COLORS[g], label=g, alpha=0.75) for g in G_ORDER]
ax_e.legend(handles=handles_e, fontsize=8, loc='upper left', ncol=2).get_frame().set_edgecolor('#ccc')
ax_e.text(-0.10, 1.04, 'E', transform=ax_e.transAxes, fontsize=PS, fontweight='bold')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 제목 & 저장                          ║
# ╚══════════════════════════════════════════════════════════════╝

fig.suptitle('Joint Effects of PC6 and Body Composition',
             fontsize=13.5, fontweight='bold', y=0.98)

plt.savefig(OUT_TIFF, dpi=300, format='tiff', bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n저장 완료: {OUT_TIFF}")