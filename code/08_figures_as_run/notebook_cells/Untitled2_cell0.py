"""
=============================================================================
Figure 3. Identification and Biological Prioritization of the
          Key Prognostic Latent Feature
=============================================================================
Panel A: Forest plot — HR ± 95%CI, TCGA + CPTAC-3 (PC1–PC10)
Panel B: Pathway enrichment heatmap across all 10 PCs

Supplementary: Bilateral significance scatter (old Panel B)

입력: TCGA_with_genes.csv, External_with_genes.csv,
      ae_latent_pca10_all.csv, ae_latent_pca10_external.csv,
      ae_model_bundle.pt
출력: Figure3.tiff, FigS_bilateral_scatter.tiff
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from lifelines import CoxPHFitter
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
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
AE_BUNDLE = os.path.join(BASE, "ae_v3_outputs", "ae_model_bundle.pt")
OUTPATH   = os.path.join(BASE, "Figure3.tiff")
OUTPATH_S = os.path.join(BASE, "FigS_bilateral_scatter.tiff")

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
    return f"HR={hr:.3f} ({lo:.3f}\u2013{hi:.3f})  {star(p)}"

def choose_selected_pc(screen_df):
    df = screen_df.copy()
    df['t_sig']       = df['t_p'] < 0.05
    df['e_sig']       = df['e_p'] < 0.05
    df['same_dir']    = [same_direction(t, e) for t, e in zip(df['t_hr'], df['e_hr'])]
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
# PART 1: LOAD DATA & COX SCREENING
# ============================================================
print("Loading clinical data...")
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
    selected_idx = SCREEN.index[SCREEN['PC'] == selected_pc][0]
    sel_row      = SCREEN[SCREEN['PC'] == selected_pc].iloc[0]

SIG = -np.log10(0.05)
print(f"\nSelected PC: {selected_pc}")

# ============================================================
# PART 2: PATHWAY ENRICHMENT (all 10 PCs)
# ============================================================
print("\nLoading autoencoder for pathway enrichment...")
bundle = torch.load(AE_BUNDLE, map_location='cpu', weights_only=False)

state_dict = bundle['state_dict']
gene_names = bundle['selected_genes']
pca_components = bundle['pca10_components']

W1 = state_dict['encoder.0.weight'].numpy()
W2 = state_dict['encoder.4.weight'].numpy()
W3 = state_dict['encoder.8.weight'].numpy()

all_pc_weights = {}
for pc_idx in range(10):
    pc_loading = pca_components[pc_idx, :]
    all_pc_weights[f'PC{pc_idx+1}'] = pc_loading @ W3 @ W2 @ W1

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

# Enrichment
print("Computing pathway enrichment...")
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
fdr_list = []
for pc in [f'PC{i}' for i in range(1, 11)]:
    sub = enrich_df[enrich_df['PC'] == pc].copy()
    _, fdr, _, _ = multipletests(sub['p_value'], method='fdr_bh')
    sub['FDR'] = fdr
    fdr_list.append(sub)
enrich_df = pd.concat(fdr_list, ignore_index=True)
print("  Done.")

# ============================================================
# FIGURE 3: Panel A (Forest) + Panel B (Heatmap)
# ============================================================
print("\nDrawing Figure 3...")

fig = plt.figure(figsize=(24, 12))
fig.patch.set_facecolor('white')

gs = GridSpec(1, 2, figure=fig,
              wspace=0.35, left=0.05, right=0.97,
              top=0.88, bottom=0.08,
              width_ratios=[1.1, 1.3])

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
        ax_a.axhspan(y-0.46, y+0.46, color='#F8F9FA', alpha=0.4, zorder=0)

    # TCGA
    tc = '#2980B9' if t_s else '#AED6F1'
    ax_a.errorbar(row['t_hr'], y+OFF,
                  xerr=[[row['t_hr']-row['t_lo']], [row['t_hi']-row['t_hr']]],
                  fmt='D' if t_s else 'o', color=tc,
                  markersize=11 if (t_s and is_sel) else (9 if t_s else 5),
                  elinewidth=2 if t_s else 1.0, capsize=4, zorder=4,
                  markeredgecolor='#111' if t_s else tc,
                  markeredgewidth=0.9 if t_s else 0)

    # CPTAC-3
    ec = '#E74C3C' if e_s else '#F1948A'
    ax_a.errorbar(row['e_hr'], y-OFF,
                  xerr=[[row['e_hr']-row['e_lo']], [row['e_hi']-row['e_hr']]],
                  fmt='s' if e_s else 'o', color=ec,
                  markersize=11 if (e_s and is_sel) else (9 if e_s else 5),
                  elinewidth=2 if e_s else 1.0, capsize=4, zorder=4,
                  markeredgecolor='#111' if e_s else ec,
                  markeredgewidth=0.9 if e_s else 0)

    ax_a.text(-0.02, y, row['PC'], va='center', ha='right', fontsize=11,
              fontweight='bold' if is_sel else 'normal',
              color='#C0392B' if is_sel else '#2C3E50',
              transform=ax_a.get_yaxis_transform())
    ax_a.text(2.90, y+OFF,
              fmt_hr(row['t_hr'], row['t_lo'], row['t_hi'], row['t_p']),
              va='center', ha='left', fontsize=8.5,
              color='#2980B9' if t_s else '#CCC',
              fontweight='bold' if t_s else 'normal')
    ax_a.text(2.90, y-OFF,
              fmt_hr(row['e_hr'], row['e_lo'], row['e_hi'], row['e_p']),
              va='center', ha='left', fontsize=8.5,
              color='#E74C3C' if e_s else '#CCC',
              fontweight='bold' if e_s else 'normal')
    if is_sel:
        for yo, co in [(OFF, '#2980B9'), (-OFF, '#E74C3C')]:
            ax_a.text(0.03, y+yo, '\u2605', va='center', ha='left',
                      fontsize=11, color=co, zorder=5,
                      transform=ax_a.get_yaxis_transform())

ax_a.set_xlim(0.45, 2.85); ax_a.set_ylim(-0.8, n-0.15); ax_a.set_yticks([])
ax_a.set_xlabel('Hazard Ratio  (95% CI)', fontsize=12)
ax_a.set_title('Prognostic Screening of Latent Features\n'
               'Multivariable Cox  |  adj: SMA, age, sex, stage  |  strat: cancer type',
               fontsize=11, fontweight='bold')
ax_a.text(2.88, n-0.42, 'HR (95% CI)   sig',
          va='center', ha='left', fontsize=9, fontweight='bold', color='#555')

leg_a = [
    Line2D([0],[0], marker='D', color='w', markerfacecolor='#2980B9',
           markeredgecolor='#111', markersize=10, label='TCGA  p<0.05'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#E74C3C',
           markeredgecolor='#111', markersize=10, label='CPTAC-3  p<0.05'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#BDC3C7',
           markersize=7, label='ns'),
]
if selected_pc:
    leg_a.append(mpatches.Patch(facecolor='#FADBD8', alpha=0.6,
                                edgecolor='#C0392B', label=f'{selected_pc}  (selected)'))
ax_a.legend(handles=leg_a, fontsize=9, framealpha=0.9, loc='lower right')
for sp in ['top', 'right', 'left']: ax_a.spines[sp].set_visible(False)
ax_a.text(-0.06, 1.05, 'A', transform=ax_a.transAxes, fontsize=16, fontweight='bold')

# ──────────────────────────────────────────────────────────────
# Panel B: Pathway Enrichment Heatmap
# ──────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

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

vmax = np.abs(heatmap_data).max()
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax_b.imshow(heatmap_data, cmap='RdBu_r', norm=norm, aspect='auto')

ax_b.set_xticks(range(len(pcs_order)))
ax_b.set_xticklabels(pcs_order, fontsize=10, fontweight='bold')
ax_b.set_yticks(range(len(pathways_order)))
ax_b.set_yticklabels(pathways_order, fontsize=9)

# Highlight selected PC column
if selected_pc and selected_pc in pcs_order:
    sel_j = pcs_order.index(selected_pc)
    ax_b.add_patch(plt.Rectangle((sel_j-0.5, -0.5), 1, len(pathways_order),
                                  linewidth=3, edgecolor='#C0392B',
                                  facecolor='none', zorder=10))

# Significance markers
for i in range(len(pathways_order)):
    for j in range(len(pcs_order)):
        row = enrich_df[(enrich_df['PC'] == pcs_order[j]) &
                        (enrich_df['Pathway'] == pathways_order[i])]
        if len(row) > 0 and row.iloc[0]['FDR'] < 0.001:
            ax_b.text(j, i, '**', ha='center', va='center',
                      fontsize=12, fontweight='bold', color='black')
        elif len(row) > 0 and row.iloc[0]['FDR'] < 0.05:
            ax_b.text(j, i, '*', ha='center', va='center',
                      fontsize=14, fontweight='bold', color='black')

cbar = plt.colorbar(im, ax=ax_b, shrink=0.7, pad=0.02,
                     label='Signed \u2212log\u2081\u2080(FDR)')
ax_b.set_title('Biological Characterization of Latent Features\n'
               'Hallmark Pathway Enrichment  |  Red=\u2191risk  Blue=\u2193protective\n'
               '* FDR<0.05  ** FDR<0.001',
               fontsize=11, fontweight='bold')
ax_b.text(-0.08, 1.05, 'B', transform=ax_b.transAxes, fontsize=16, fontweight='bold')

# ── Figure title ──
fig.suptitle('Identification and Biological Prioritization of the Key Prognostic Latent Feature',
             fontsize=14, fontweight='bold', y=0.96)

plt.savefig(OUTPATH, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
plt.close()
print(f"\u2713 Saved: {OUTPATH}")

# ============================================================
# SUPPLEMENTARY: Bilateral Significance Scatter (old Panel B)
# ============================================================
print("\nDrawing supplementary figure...")

fig_s, ax_s = plt.subplots(figsize=(10, 10))
fig_s.patch.set_facecolor('white')

t_lp = [safe_neglog10(p) for p in SCREEN['t_p']]
e_lp = [safe_neglog10(p) for p in SCREEN['e_p']]
eff  = np.abs(SCREEN['t_hr'].values-1) + np.abs(SCREEN['e_hr'].values-1)
eff  = np.nan_to_num(eff, nan=0.0)
sz   = eff * 700 + 60

xmax = max(max(t_lp)*1.5, SIG+0.8, 2.0)
ymax = max(max(e_lp)*1.9, SIG+0.8, 2.0)

ax_s.fill_between([SIG, xmax], SIG, ymax, color='#FADBD8', alpha=0.25, zorder=0)
ax_s.axvline(SIG, color='#2980B9', lw=1.2, ls='--', alpha=0.65)
ax_s.axhline(SIG, color='#E74C3C', lw=1.2, ls='--', alpha=0.65)
ax_s.text(SIG+0.04, 0.05, 'TCGA p=0.05',
          color='#2980B9', fontsize=8.5, rotation=90, va='bottom')
ax_s.text(0.05, SIG+0.04, 'CPTAC-3 p=0.05',
          color='#E74C3C', fontsize=8.5, va='bottom')

for i, row in SCREEN.iterrows():
    is_sel = bool(selected_pc and row['PC'] == selected_pc)
    t_s = row['t_sig']; e_s = row['e_sig']; bil = row['bilateral']
    if is_sel:            col = '#C0392B'
    elif bil:             col = '#27AE60'
    elif t_s and not e_s: col = '#2980B9'
    elif e_s and not t_s: col = '#9B59B6'
    else:                 col = '#BDC3C7'

    ax_s.scatter(t_lp[i], e_lp[i], s=sz[i], c=col,
                 zorder=4 if is_sel else 3,
                 edgecolors='#111' if is_sel else col,
                 linewidths=1.8 if is_sel else 0.4, alpha=0.90)
    ax_s.annotate(row['PC'],
                  xy=(t_lp[i], e_lp[i]),
                  xytext=(t_lp[i]+0.06, e_lp[i]+0.06),
                  fontsize=10.5 if is_sel else 9.5,
                  fontweight='bold' if is_sel else 'normal',
                  color='#C0392B' if is_sel else '#444')

if selected_pc and sel_row is not None:
    direction_text = "same direction" if sel_row['same_dir'] else "discordant"
    ax_s.annotate(
        (f'\u2605  {selected_pc}\n'
         f'TCGA p={sel_row["t_p"]:.3f} {star(sel_row["t_p"])}\n'
         f'CPTAC-3 p={sel_row["e_p"]:.3f} {star(sel_row["e_p"])}\n'
         f'\u2192 Bilateral + {direction_text}'),
        xy=(t_lp[selected_idx], e_lp[selected_idx]),
        xytext=(t_lp[selected_idx]+0.40, e_lp[selected_idx]+0.55),
        fontsize=9.5, color='#C0392B', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.30', facecolor='#FADBD8',
                  edgecolor='#C0392B', alpha=0.92),
        arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.8))

ax_s.set_xlabel('\u2212log\u2081\u2080(p)  TCGA', fontsize=12)
ax_s.set_ylabel('\u2212log\u2081\u2080(p)  CPTAC-3', fontsize=12)
ax_s.set_xlim(0, xmax); ax_s.set_ylim(0, ymax)
ax_s.set_title('Bilateral Prognostic Significance\nTCGA vs CPTAC-3',
               fontsize=12, fontweight='bold')
for sp in ['top', 'right']: ax_s.spines[sp].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPATH_S, dpi=300, bbox_inches='tight', facecolor='white', format='tiff')
plt.close()
print(f"\u2713 Saved: {OUTPATH_S}")

print("\n\u2713 All done!")