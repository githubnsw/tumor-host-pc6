"""
=============================================================================
Biological Characterization of All 10 Latent Components
=============================================================================
1. Extract encoder weights → gene-to-PC weight for ALL 10 PCs
2. Pathway enrichment (Hallmark, Wilcoxon rank-sum) for ALL 10 PCs
3. Summary heatmap: PC × Pathway
4. Detailed volcano/bar for prognostically significant PC
5. Correlation validation
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
import torch
from scipy.stats import spearmanr, ranksums
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AUTOENCODER MODEL
# =============================================================================
# ★★ 경로를 본인 환경에 맞게 수정하세요 ★★
bundle = torch.load(r'C:\Users\nsw\Desktop\body_composition\ae_v3_outputs\ae_model_bundle.pt',
                    map_location='cpu', weights_only=False)

state_dict = bundle['state_dict']
gene_names = bundle['selected_genes']
pca_components = bundle['pca10_components']  # (10, 64)

# Encoder: Linear(0) → BN(1) → ReLU → Dropout → Linear(4) → BN(5) → ReLU → Dropout → Linear(8)
W1 = state_dict['encoder.0.weight'].numpy()   # (1024, 3000)
W2 = state_dict['encoder.4.weight'].numpy()   # (256, 1024)
W3 = state_dict['encoder.8.weight'].numpy()   # (64, 256)

print(f"Genes: {len(gene_names)}")
print(f"Encoder: {W1.shape[1]} → {W1.shape[0]} → {W2.shape[0]} → {W3.shape[0]} → PCA(10)")

# =============================================================================
# 2. GENE → PC WEIGHT FOR ALL 10 PCs
# =============================================================================
all_pc_weights = {}

for pc_idx in range(10):
    pc_loading = pca_components[pc_idx, :]
    weights = pc_loading @ W3 @ W2 @ W1
    all_pc_weights[f'PC{pc_idx+1}'] = weights

# Save full weight matrix
weight_df = pd.DataFrame(all_pc_weights, index=gene_names)
weight_df.index.name = 'gene'
weight_df.to_csv('gene_weights_all_PCs.csv')
print(f"\n✓ Gene weights for all 10 PCs saved: gene_weights_all_PCs.csv")

# Top genes per PC
print(f"\n{'='*80}")
print("TOP 5 RISK / PROTECTIVE GENES PER PC")
print(f"{'='*80}")
for pc in [f'PC{i}' for i in range(1, 11)]:
    w = weight_df[pc]
    top_risk = w.nlargest(5)
    top_prot = w.nsmallest(5)
    risk_str = ', '.join(f'{g}({v:+.2f})' for g, v in top_risk.items())
    prot_str = ', '.join(f'{g}({v:+.2f})' for g, v in top_prot.items())
    print(f"\n  {pc}:")
    print(f"    Risk↑:  {risk_str}")
    print(f"    Prot↓:  {prot_str}")

# =============================================================================
# 3. HALLMARK GENE SETS
# =============================================================================
hallmark_sets = {
    'Glycolysis': ['PKM','ENO1','SLC16A3','PFKP','PGAM1','LDHA','AKR1B1','GAPDH',
                   'GPI','TPI1','ALDOA','PGK1','HK2','HK1','PFKL','ENO2','LDHB',
                   'SLC2A1','SLC2A3','BPGM','PCK1','PCK2','FBP1','ALDOB','ALDOC'],
    'Hypoxia': ['NDRG1','VEGFA','LOX','BNIP3','ANGPTL4','SLC2A1','ADM','P4HA1',
                'DDIT4','EGLN3','PDK1','LDHA','PGK1','HK2','ENO1','ALDOA','SLC2A3',
                'CA9','BNIP3L','IGFBP3','ANKRD37','PLOD2','MXI1','ERO1A','PFKFB3'],
    'IFN-alpha Response': ['ISG15','MX1','IFIT1','IFIT3','OAS1','OAS2','OAS3','STAT1',
                           'IRF7','IFI44','IFI44L','RSAD2','IFI6','HERC5','USP18',
                           'IFITM1','BST2','XAF1','TRIM22','DDX58','CXCL10','GBP1'],
    'IFN-gamma Response': ['GBP1','STAT1','IRF1','CXCL9','CXCL10','CXCL11','IDO1',
                           'TAP1','TAP2','PSMB9','PSMB8','HLA-A','HLA-B','HLA-C',
                           'B2M','CD274','CIITA','GBP2','GBP4','GBP5','WARS1'],
    'Cholesterol Homeostasis': ['PCSK9','INSIG1','HMGCR','HMGCS1','FDFT1','SQLE',
                                 'MVD','MVK','IDI1','FDPS','LSS','DHCR7','DHCR24',
                                 'SC5D','CYP51A1','MSMO1','NSDHL','HSD17B7','EBP',
                                 'LDLR','SREBF2','ACAT2','ACLY','SCD'],
    'mTORC1 Signaling': ['SLC7A5','SLC3A2','SLC1A5','MTHFD2','SHMT2','PSAT1','PHGDH',
                          'PSPH','CBS','CTH','SRM','ODC1','AMD1','MTOR','RPTOR',
                          'EIF4EBP1','RPS6KB1','DDIT4','SESN2','TRIB3','NUPR1'],
    'p53 Pathway': ['CDKN1A','MDM2','BAX','BBC3','PMAIP1','GADD45A','FAS','TNFRSF10B',
                    'TP53INP1','SESN1','SESN2','DDIT4','PLK2','PLK3','CCNG1','DDB2',
                    'XPC','FDXR','RRM2B','TIGAR','ZMAT3','PHLDA3'],
    'TNF-alpha/NF-kB': ['TNFAIP3','NFKBIA','NFKB1','RELA','RELB','BCL3','CXCL1',
                         'CXCL2','CXCL3','CCL2','CCL5','IL6','IL8','ICAM1','VCAM1',
                         'TRAF1','BIRC3','BIRC2','CFLAR','TNFAIP2','TNFAIP8'],
    'EMT': ['VIM','CDH2','SNAI1','SNAI2','TWIST1','ZEB1','ZEB2','FN1',
            'SPARC','MMP2','MMP9','COL1A1','COL1A2','COL3A1','COL5A1',
            'ACTA2','TAGLN','POSTN','LOX','LOXL2','SERPINE1','ITGA5'],
    'Angiogenesis': ['VEGFA','VEGFB','VEGFC','FLT1','KDR','FLT4','ANGPT1','ANGPT2',
                     'TEK','PECAM1','CDH5','ENG','PDGFB','PDGFRB','NRP1','NRP2',
                     'HIF1A','EPAS1','DLL4','NOTCH1','JAG1','THBS1'],
    'Inflammatory Response': ['IL6','IL1B','IL1A','TNF','CXCL8','CCL2','CCL3','CCL4',
                               'CXCL1','CXCL2','CXCL3','PTGS2','SELE','ICAM1','VCAM1',
                               'NFKB1','TLR2','TLR4','MYD88','IRAK4','TRAF6'],
    'Bile Acid Metabolism': ['CYP7A1','CYP8B1','CYP27A1','BAAT','SLC27A5','ABCB11',
                              'NR1H4','SLCO1B1','SLCO1B3','SLC10A1','SLC10A2',
                              'AKR1D1','HSD3B7','AMACR','ACOX2','SCP2','UGT2B4'],
    'Xenobiotic Metabolism': ['CYP1A1','CYP1A2','CYP1B1','CYP2B6','CYP2C9','CYP2C19',
                               'CYP2D6','CYP2E1','CYP3A4','CYP3A5','UGT1A1','UGT1A6',
                               'GSTA1','GSTA2','GSTM1','GSTP1','GSTT1','NR1I2','NR1I3',
                               'EPHX1','ALDH3A1','AKR1C1','AKR1C2','AKR1C3'],
    'Fatty Acid Metabolism': ['ACACA','FASN','SCD','ELOVL6','FADS1','FADS2','ACSL1',
                               'ACSL3','ACSL4','ACSL5','CPT1A','CPT2','ACADM','ACADL',
                               'ACADVL','HADHA','HADHB','ECHS1','EHHADH','ACOX1'],
    'Oxidative Phosphorylation': ['NDUFA1','NDUFA2','NDUFB1','NDUFS1','NDUFS2',
                                   'SDHA','SDHB','UQCRC1','UQCRC2','CYC1',
                                   'COX5A','COX5B','COX6A1','COX7A2','ATP5F1A',
                                   'ATP5F1B','ATP5F1C','ATP5MC1','ATP5PD'],
    'Apoptosis': ['BAX','BAK1','BID','BIM','BAD','CASP3','CASP7','CASP8','CASP9',
                  'CYCS','APAF1','DIABLO','XIAP','BCL2','BCL2L1','MCL1','TNFRSF10A',
                  'TNFRSF10B','FAS','FASLG','TRAIL'],
    'DNA Repair': ['BRCA1','BRCA2','RAD51','ATM','ATR','CHEK1','CHEK2','TP53',
                   'XRCC1','PARP1','MLH1','MSH2','MSH6','PMS2','ERCC1','ERCC2',
                   'XPA','XPC','OGG1','MUTYH','MGMT'],
    'Cell Cycle (G2M)': ['CDK1','CCNB1','CCNB2','CCNA2','CDC20','CDC25C','PLK1',
                          'AURKA','AURKB','BUB1','BUB1B','MAD2L1','TTK','NDC80',
                          'CENPA','CENPF','TOP2A','MKI67','KIF11','KIF23','BIRC5'],
    'Complement': ['C1QA','C1QB','C1QC','C2','C3','C4A','C4B','C5','C6','C7',
                   'C8A','C8B','C9','CFB','CFD','CFH','CFI','MASP1','MASP2','MBL2'],
}

# =============================================================================
# 4. PATHWAY ENRICHMENT FOR ALL 10 PCs
# =============================================================================
print(f"\n{'='*90}")
print("PATHWAY ENRICHMENT — ALL 10 PCs (Wilcoxon rank-sum)")
print(f"{'='*90}")

weight_dict_all = {}
for pc in [f'PC{i}' for i in range(1, 11)]:
    weight_dict_all[pc] = dict(zip(gene_names, all_pc_weights[pc]))

all_enrich = []
for pc in [f'PC{i}' for i in range(1, 11)]:
    wd = weight_dict_all[pc]
    for pathway, genes in hallmark_sets.items():
        present = [g for g in genes if g in wd]
        if len(present) < 5:
            continue
        in_w = np.array([wd[g] for g in present])
        out_w = np.array([wd[g] for g in gene_names if g not in present])
        stat, pval = ranksums(in_w, out_w)
        all_enrich.append({
            'PC': pc, 'Pathway': pathway,
            'N_genes': len(present), 'Mean_weight': in_w.mean(),
            'Statistic': stat, 'p_value': pval,
        })

enrich_df = pd.DataFrame(all_enrich)

# FDR per PC
fdr_list = []
for pc in [f'PC{i}' for i in range(1, 11)]:
    sub = enrich_df[enrich_df['PC'] == pc].copy()
    _, fdr, _, _ = multipletests(sub['p_value'], method='fdr_bh')
    sub['FDR'] = fdr
    fdr_list.append(sub)
enrich_df = pd.concat(fdr_list, ignore_index=True)

enrich_df.to_csv('pathway_enrichment_all_PCs.csv', index=False)
print(f"✓ Saved: pathway_enrichment_all_PCs.csv")

# Print significant per PC
for pc in [f'PC{i}' for i in range(1, 11)]:
    sub = enrich_df[(enrich_df['PC'] == pc) & (enrich_df['FDR'] < 0.05)]
    sub = sub.sort_values('Mean_weight', ascending=False)
    if len(sub) == 0:
        print(f"\n  {pc}: No pathways FDR<0.05")
    else:
        print(f"\n  {pc}: {len(sub)} pathways FDR<0.05")
        for _, r in sub.iterrows():
            direction = '↑Risk' if r['Mean_weight'] > 0 else '↓Protective'
            print(f"    {r['Pathway']:30s} mean_w={r['Mean_weight']:+.3f} FDR={r['FDR']:.4f} {direction}")

# =============================================================================
# 5. HEATMAP: PC × Pathway (signed -log10 FDR)
# =============================================================================
pathways_order = sorted(hallmark_sets.keys())
pcs_order = [f'PC{i}' for i in range(1, 11)]

heatmap_data = np.zeros((len(pathways_order), len(pcs_order)))
for i, pathway in enumerate(pathways_order):
    for j, pc in enumerate(pcs_order):
        row = enrich_df[(enrich_df['PC'] == pc) & (enrich_df['Pathway'] == pathway)]
        if len(row) == 0:
            heatmap_data[i, j] = 0
        else:
            fdr = row.iloc[0]['FDR']
            mean_w = row.iloc[0]['Mean_weight']
            val = -np.log10(max(fdr, 1e-10))
            heatmap_data[i, j] = val if mean_w > 0 else -val

fig, ax = plt.subplots(figsize=(14, 10))

vmax = np.abs(heatmap_data).max()
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(heatmap_data, cmap='RdBu_r', norm=norm, aspect='auto')

ax.set_xticks(range(len(pcs_order)))
ax.set_xticklabels(pcs_order, fontsize=10, fontweight='bold')
ax.set_yticks(range(len(pathways_order)))
ax.set_yticklabels(pathways_order, fontsize=9)

for i in range(len(pathways_order)):
    for j in range(len(pcs_order)):
        row = enrich_df[(enrich_df['PC'] == pcs_order[j]) &
                        (enrich_df['Pathway'] == pathways_order[i])]
        if len(row) > 0 and row.iloc[0]['FDR'] < 0.001:
            ax.text(j, i, '**', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='black')
        elif len(row) > 0 and row.iloc[0]['FDR'] < 0.05:
            ax.text(j, i, '*', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='black')

cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Signed -log₁₀(FDR)')
ax.set_title('Pathway Enrichment Across All Latent Components\n'
             'Red = Positive weight (↑risk) | Blue = Negative weight (↓protective)\n'
             '* FDR<0.05  ** FDR<0.001',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_heatmap_all_PCs.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("\n✓ Heatmap saved: fig_heatmap_all_PCs.png")

# =============================================================================
# 6. CORRELATION VALIDATION (all 10 PCs)
# =============================================================================
print(f"\n{'='*90}")
print("VALIDATION: Encoder Weight vs Expression Correlation (all PCs)")
print(f"{'='*90}")

ae_pca = pd.read_csv(r'C:\Users\nsw\Desktop\body_composition\ae_latent_pca10_all.csv')

g1 = pd.read_csv(r'C:\Users\nsw\Desktop\body_composition\RNA_Group1_updated_to_new_symbol.csv')
g2 = pd.read_csv(r'C:\Users\nsw\Desktop\body_composition\RNA_Group2_updated_to_new_symbol.csv')
g3 = pd.read_csv(r'C:\Users\nsw\Desktop\body_composition\RNA_Group3_updated_to_new_symbol.csv')
g4 = pd.read_csv(r'C:\Users\nsw\Desktop\body_composition\RNA_Group4_updated_to_new_symbol.csv')
rna = g1.set_index('gene')
for g in [g2, g3, g4]:
    rna = rna.join(g.set_index('gene'), how='outer')

common_samples = list(set(rna.columns) & set(ae_pca['sample_id']))
rna = rna[common_samples]
print(f"Samples: {len(common_samples)}")

for pc_idx in range(10):
    pc_name = f'PC{pc_idx+1}'
    ae_col = f'AE_{pc_name}'
    pc_vals = ae_pca.set_index('sample_id')[ae_col][common_samples].values
    pc_w = all_pc_weights[pc_name]

    corrs = []
    for gi, gene in enumerate(gene_names):
        if gene not in rna.index:
            continue
        expr = rna.loc[gene, common_samples].astype(float).values
        valid = ~(np.isnan(expr) | np.isnan(pc_vals))
        if valid.sum() > 50:
            r, _ = spearmanr(pc_vals[valid], expr[valid])
            corrs.append(r)
        else:
            corrs.append(np.nan)

    corr_arr = np.array(corrs)
    weight_arr = pc_w[:len(corrs)]
    valid_both = ~(np.isnan(corr_arr))
    rho, p = spearmanr(weight_arr[valid_both], corr_arr[valid_both])
    print(f"  {pc_name}: Weight vs Corr concordance rho={rho:.3f} (p={p:.1e})")

# =============================================================================
# 7. DETAILED FIGURES FOR PROGNOSTIC PC (PC6)
# =============================================================================
print(f"\n{'='*90}")
print("DETAILED FIGURES — PC6 (Prognostically Significant)")
print(f"{'='*90}")

pc6_w = weight_df['PC6']
pc6_vals = ae_pca.set_index('sample_id')['AE_PC6'][common_samples].values

# Correlation for PC6
corr_results = []
for gene in gene_names:
    if gene not in rna.index:
        continue
    expr = rna.loc[gene, common_samples].astype(float).values
    valid = ~(np.isnan(expr) | np.isnan(pc6_vals))
    if valid.sum() > 50:
        r, p = spearmanr(pc6_vals[valid], expr[valid])
        corr_results.append({'gene': gene, 'corr': r, 'p': p, 'weight': pc6_w[gene]})

corr_df = pd.DataFrame(corr_results)

# --- Figure A: Pathway enrichment bar (PC6 only) ---
pc6_enrich = enrich_df[enrich_df['PC'] == 'PC6'].sort_values('Mean_weight')
sig_pc6 = pc6_enrich[pc6_enrich['FDR'] < 0.1]

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#1565C0' if w < 0 else '#D32F2F' for w in sig_pc6['Mean_weight']]
ax.barh(range(len(sig_pc6)), sig_pc6['Mean_weight'], color=colors, edgecolor='white')
ax.set_yticks(range(len(sig_pc6)))
ax.set_yticklabels(sig_pc6['Pathway'], fontsize=10)
for i, (_, r) in enumerate(sig_pc6.iterrows()):
    sig = '***' if r['FDR'] < 0.001 else '**' if r['FDR'] < 0.01 else '*' if r['FDR'] < 0.05 else '†'
    offset = 0.02 if r['Mean_weight'] > 0 else -0.02
    ha = 'left' if r['Mean_weight'] > 0 else 'right'
    ax.text(r['Mean_weight'] + offset, i, f"FDR={r['FDR']:.4f} {sig}",
            va='center', ha=ha, fontsize=8)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Mean Gene-to-PC6 Weight', fontsize=11, fontweight='bold')
ax.set_title('Pathway Enrichment — Prognostic Latent Feature\n'
             'Red = Risk (score↑) | Blue = Protective (score↓)',
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('fig_pc6_pathway.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("✓ PC6 pathway bar saved")

# --- Figure B: Volcano plot (PC6) ---
fig, ax = plt.subplots(figsize=(10, 6))
corr_df['neg_log10_p'] = -np.log10(corr_df['p'].clip(lower=1e-300))
ax.scatter(corr_df['weight'], corr_df['neg_log10_p'], alpha=0.3, s=15, color='#9E9E9E')

top_risk = corr_df.nlargest(10, 'weight')
top_prot = corr_df.nsmallest(10, 'weight')
ax.scatter(top_risk['weight'], top_risk['neg_log10_p'], color='#D32F2F', s=50, zorder=5, edgecolors='white')
ax.scatter(top_prot['weight'], top_prot['neg_log10_p'], color='#1565C0', s=50, zorder=5, edgecolors='white')

for _, r in pd.concat([top_risk.head(5), top_prot.head(5)]).iterrows():
    ax.annotate(r['gene'], (r['weight'], r['neg_log10_p']),
                fontsize=7, fontweight='bold', xytext=(5, 5), textcoords='offset points')

ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Gene-to-PC6 Encoder Weight', fontsize=11, fontweight='bold')
ax.set_ylabel('-log₁₀(p-value)', fontsize=11, fontweight='bold')
ax.set_title('Gene Contributions to Prognostic Latent Feature',
             fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('fig_pc6_volcano.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("✓ PC6 volcano saved")

# --- Figure C: Weight vs Correlation scatter (PC6) ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(corr_df['weight'], corr_df['corr'], alpha=0.2, s=15, color='#455A64')

pathway_colors = {
    'Glycolysis': '#D32F2F', 'Hypoxia': '#FF6F00',
    'Cholesterol Homeostasis': '#1565C0', 'mTORC1 Signaling': '#0D47A1',
}
for pathway, color in pathway_colors.items():
    genes_in = [g for g in hallmark_sets[pathway] if g in corr_df['gene'].values]
    sub = corr_df[corr_df['gene'].isin(genes_in)]
    ax.scatter(sub['weight'], sub['corr'], alpha=0.7, s=40, color=color,
               edgecolors='white', linewidth=0.5, label=pathway, zorder=5)

rho, p = spearmanr(corr_df['weight'], corr_df['corr'])
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Encoder Weight (Gene → PC)', fontsize=11, fontweight='bold')
ax.set_ylabel('Spearman Correlation (Gene ~ PC Score)', fontsize=11, fontweight='bold')
ax.set_title(f'Weight vs Correlation Concordance\nrho={rho:.3f} (p={p:.1e})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('fig_pc6_weight_vs_corr.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("✓ Weight vs correlation scatter saved")

print("\n✓ All done!")