"""
Figure 6. Joint Effects of PC6 and Body Composition  (Standalone)
==================================================================
수정사항 (이번 차):
  ▶ Panel C (GSEA) 제거
  ▶ Panel D (Heatmap) → Panel C 로 재배치
  ▶ Panel E (Violin) → Panel D 로 재배치
  ▶ Layout: 4 panels (A, B 위 / C, D 아래) — 2x2 그리드
  ▶ Panel B: 겹치는 annotation 제거
    - HAVCR2, MSTN, MRC1 등 q=0.05 임계선 근처에 몰린 유전자 제외
    - 깔끔하게 떨어진 위쪽 유전자만 표시 (CA9, NDRG1, SLC16A3, VEGFA, ALDOA, PKM)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import kruskal, ranksums
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = r"C:\Users\nsw\Desktop\body_composition"

TCGA_FILE    = os.path.join(BASE_DIR, "TCGA_with_genes.csv")
PC_TCGA_FILE = os.path.join(BASE_DIR, "ae_latent_pca10_all.csv")
TCGA_RNA_FILES = [
    os.path.join(BASE_DIR, f"RNA_Group{i}_updated_to_new_symbol.csv")
    for i in range(1, 5)
]
OUT_TIFF = os.path.join(BASE_DIR, "Figure6.tiff")

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
G_SHORT  = {'PC6-Low / SMA-High' :'PC6-L / SMA-H',
            'PC6-Low / SMA-Low'  :'PC6-L / SMA-L',
            'PC6-High / SMA-High':'PC6-H / SMA-H',
            'PC6-High / SMA-Low' :'PC6-H / SMA-L'}
G_COLORS = {'PC6-Low / SMA-High':'#2166AC','PC6-Low / SMA-Low':'#74ADD1',
            'PC6-High / SMA-High':'#F46D43','PC6-High / SMA-Low':'#D73027'}

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


print("Computing heatmap z-scores + KW test...")

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
sig_pvals = {}
for sig_name, genes in SIGS.items():
    avail = [g for g in genes if g in rna_log.index]
    if not avail:
        continue
    row, all_vals = {}, []
    grp_arrays = []
    for grp in G_ORDER:
        pids = [p for p in merged[merged['group4']==grp]['pid'] if p in rna_log.columns]
        val  = rna_log.loc[avail, pids].values.mean()
        row[grp] = val
        all_vals.append(val)
        grp_arrays.append(rna_log.loc[avail, pids].values.flatten())
    mu, sd = np.mean(all_vals), np.std(all_vals)
    for grp in G_ORDER:
        row[grp] = (row[grp] - mu) / (sd + 1e-9)
    hm_rows[sig_name] = row
    try:
        _, kp = kruskal(*grp_arrays)
        sig_pvals[sig_name] = kp
    except:
        sig_pvals[sig_name] = np.nan

heatmap_z = pd.DataFrame(hm_rows).T

def star_str(p):
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 레이아웃 (2x2)                       ║
# ╚══════════════════════════════════════════════════════════════╝

print("Drawing figure...")
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('white')
gs = GridSpec(2, 2, figure=fig,
              hspace=0.35,
              wspace=0.30,
              left=0.07, right=0.96,
              top=0.88, bottom=0.05,            # ← 0.93 → 0.88 (suptitle 공간 확보)
              height_ratios=[1.15, 1.0])

LS, TS, TIS, PS = 18, 16, 19, 22
TITLE_PAD = 12

def add_panel_label(ax, letter):
    ax.set_title(letter, loc='left', fontsize=PS, fontweight='bold', pad=TITLE_PAD)

# ════════════════════════════════════════════════════════════════
# ── A: KM + Number at Risk + Log-rank p ────────────────────────
# ════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs[0, 0])

for grp in G_ORDER:
    sub = merged[merged['group4']==grp]
    kmf = KaplanMeierFitter()
    kmf.fit(sub['OS_months'], sub['OS'], label=f"{G_SHORT[grp]} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax_a, color=G_COLORS[grp], lw=2.6,
                               ci_show=True, ci_alpha=0.10,
                               show_censors=True, censor_styles={'ms':5,'marker':'+'})

lrp = multivariate_logrank_test(merged['OS_months'], merged['group4'], merged['OS']).p_value
p_str = f'Log-rank  p = {lrp:.2e}' if lrp < 1e-3 else f'Log-rank  p = {lrp:.4f}'

max_t = merged['OS_months'].max()
xmax_a = int(np.ceil(max_t / 20.0)) * 20
ax_a.set_xlim(0, xmax_a)
ax_a.set_xticks(np.arange(0, xmax_a + 1, 20))

ax_a.set_xlabel("Time (months)", fontsize=LS)
ax_a.set_ylabel("Overall Survival Probability", fontsize=LS)
ax_a.tick_params(labelsize=TS)
ax_a.set_title('PC6 × SMA 4-Group Survival\n(worst = PC6-High / SMA-Low)',
               loc='center', fontsize=TIS, fontweight='bold', pad=TITLE_PAD)
add_panel_label(ax_a, 'A')

ax_a.spines[['top','right']].set_visible(False)
# left spine을 곡선 영역(y=0~1.05)에만 그림 → risk table 영역 y축 선 제거
ax_a.spines['left'].set_bounds(0, 1.05)

# 범례 — log-rank 자리(우상단)로 이동
ax_a.legend(fontsize=11, loc='upper right',
            bbox_to_anchor=(0.97, 0.97)).get_frame().set_edgecolor('#ccc')

# log-rank 박스 — 곡선 영역 우하단(data 좌표 y=0.06)으로 이동
ax_a.text(xmax_a * 0.97, 0.06, p_str,
          fontsize=14, va='bottom', ha='right',
          fontweight='bold', color='#222',
          bbox=dict(boxstyle='round,pad=0.4', fc='#FFF8E1',
                    ec='#F39C12', lw=1.5, alpha=0.95))

# ────────────────────────────────────────────────────────────
# Risk table layout (모두 DATA 좌표 사용)
#   y =  0.00       곡선/표 영역 분리선
#   y = -0.10       "Number at risk" 헤더 (표 안 상단)
#   y = -0.16       헤더 아래 구분선
#   y = -0.24~-0.48 4 그룹 행 (간격 0.08)
#   y = -0.55       표 영역 바닥
#   (axes ylim 바닥 = -0.60, 그 아래 xticklabels & xlabel)
# ────────────────────────────────────────────────────────────
risk_times = np.arange(0, xmax_a + 1, 20).astype(float)
ax_a.set_ylim(-0.60, 1.05)
ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

# 표 영역 배경
ax_a.axhspan(-0.55, -0.05, color='#F7F7F7', alpha=0.7, zorder=0)

# 곡선 / 표 분리선
ax_a.axhline(0, color='#222', lw=1.3, alpha=0.9, zorder=2)

# "Number at risk" 헤더 — DATA 좌표로 표 안쪽 상단에 (xlabel과 안 겹침)
ax_a.text(xmax_a / 2, -0.10, 'Number at risk',
          ha='center', va='center',
          fontsize=14, fontweight='bold', color='#222', zorder=3)

# 헤더 아래 구분선
ax_a.axhline(-0.16, xmin=0.005, xmax=0.995,
             color='#BBB', lw=0.7, alpha=0.8, zorder=2)

# 시간 컬럼 vertical guide (옅게)
for t in risk_times:
    ax_a.plot([t, t], [-0.54, -0.18],
              color='#E0E0E0', lw=0.5, alpha=0.6, zorder=1)

# 4개 그룹 행 (색 사각형 제거 — 텍스트 색상만으로 충분)
row_y   = [-0.24, -0.32, -0.40, -0.48]
label_x = -xmax_a * 0.04

for gi, grp in enumerate(G_ORDER):
    sub = merged[merged['group4']==grp]
    times = sub['OS_months'].values
    counts = [int(np.sum(times >= t)) for t in risk_times]

    # 그룹 라벨 (텍스트만 — 색 사각형 없음)
    ax_a.text(label_x, row_y[gi], G_SHORT[grp],
              fontsize=13, va='center', ha='right',
              fontweight='bold', color=G_COLORS[grp])

    # 위험집단 수
    for t, n in zip(risk_times, counts):
        ax_a.text(t, row_y[gi], str(n),
                  fontsize=14, va='center', ha='center',
                  color=G_COLORS[grp], fontweight='600')

# ════════════════════════════════════════════════════════════════
# ── B: Volcano (겹치는 annotation 제거) ─────────────────────────
# ════════════════════════════════════════════════════════════════
ax_b = fig.add_subplot(gs[0, 1])
up = (deg['q']<0.05) & (deg['logFC']>0.5)
dn = (deg['q']<0.05) & (deg['logFC']<-0.5)
ns_mask = ~(up|dn)

qclip = lambda s: -np.log10(s.clip(1e-300))

ax_b.scatter(deg.loc[ns_mask,'logFC'], qclip(deg.loc[ns_mask,'q']),
             c='#CCCCCC', s=4, alpha=0.4, rasterized=True)
ax_b.scatter(deg.loc[up,'logFC'], qclip(deg.loc[up,'q']),
             c='#D73027', s=12, alpha=0.75, rasterized=True, label=f'Up ({up.sum()})')
ax_b.scatter(deg.loc[dn,'logFC'], qclip(deg.loc[dn,'q']),
             c='#2166AC', s=12, alpha=0.75, rasterized=True, label=f'Down ({dn.sum()})')

ax_b_xlim_max = max(2.5, np.ceil(deg['logFC'].abs().max() * 10) / 10 + 0.1)
ax_b.set_xlim(-ax_b_xlim_max, ax_b_xlim_max)

# ── 깨끗하게 떨어진 유전자만 표시 ─────────────────────────────
# 제거: HAVCR2, MSTN, MRC1 (q=0.05 임계선 근처에 몰려 겹침)
GENE_ANNOTS = [('CA9','#D73027'),
               ('NDRG1','#D73027'),
               ('SLC16A3','#D73027'),
               ('VEGFA','#D73027'),
               ('ALDOA','#D73027'),
               ('PKM','#D73027')]
for g, col in GENE_ANNOTS:
    row = deg[deg['gene']==g]
    if len(row)==0: continue
    r = row.iloc[0]
    if r['logFC'] > 0.5:
        offset = (-6, 2); ha = 'right'
    elif r['logFC'] < -0.5:
        offset = (6, 2); ha = 'left'
    else:
        offset = (4, 2); ha = 'left'
    ax_b.annotate(g, (r['logFC'], -np.log10(max(r['q'], 1e-300))),
                  xytext=offset, textcoords='offset points',
                  ha=ha, va='bottom',
                  fontsize=13, fontweight='bold', color=col,
                  annotation_clip=True)

# FOXP3는 별도 박스로 유지 (왼쪽 영역, 잘 떨어져 있음)
foxp3 = deg[deg['gene']=='FOXP3']
if len(foxp3) > 0:
    fx = foxp3.iloc[0]['logFC']
    fy = -np.log10(max(foxp3.iloc[0]['q'], 1e-300))
    ax_b.scatter(fx, fy, c='#2166AC', s=100, zorder=5, edgecolors='white', linewidths=1.6)
    ax_b.annotate('FOXP3', xy=(fx, fy), xytext=(fx-0.55, fy+0.7),
                  fontsize=15, fontweight='bold', color='#2166AC',
                  bbox=dict(boxstyle='round,pad=0.25', fc='#EBF5FB', ec='#2166AC', alpha=0.92),
                  arrowprops=dict(arrowstyle='->', color='#2166AC', lw=1.5),
                  annotation_clip=True)

ax_b.axvline(0.5, color='gray', lw=0.7, ls='--', alpha=0.6)
ax_b.axvline(-0.5, color='gray', lw=0.7, ls='--', alpha=0.6)
ax_b.axhline(-np.log10(0.05), color='gray', lw=0.7, ls='--', alpha=0.6)
ax_b.text(ax_b_xlim_max*0.98, -np.log10(0.05)+0.03, 'q=0.05',
          ha='right', va='bottom', fontsize=11, color='gray', style='italic')

ax_b.set_xlabel("log₂FC (PC6-High/SMA-Low vs Others)", fontsize=LS)
ax_b.set_ylabel("−log₁₀(q)   (FDR-adjusted)", fontsize=LS)
ax_b.tick_params(labelsize=TS)
ax_b.set_title('DEG: PC6-High/SMA-Low vs All Others\n(FDR<0.05, |log₂FC|≥0.5)',
               loc='center', fontsize=TIS, fontweight='bold', pad=TITLE_PAD)
add_panel_label(ax_b, 'B')

ax_b.legend(fontsize=15, markerscale=2, loc='lower right').get_frame().set_edgecolor('#ccc')
ax_b.spines[['top','right']].set_visible(False)

# ════════════════════════════════════════════════════════════════
# ── C: Heatmap (이전 D, 별표 이름 바로 밑) ──────────────────────
# ════════════════════════════════════════════════════════════════
ax_c = fig.add_subplot(gs[1, 0])
hm = heatmap_z[G_ORDER]
im = ax_c.imshow(hm.values, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax_c.set_xticks(range(4))
ax_c.set_xticklabels(G_LABELS, fontsize=14)
for i, grp in enumerate(G_ORDER):
    ax_c.get_xticklabels()[i].set_color(G_COLORS[grp])
    ax_c.get_xticklabels()[i].set_fontweight('bold')

ax_c.set_yticks(range(len(hm)))
ax_c.set_yticklabels(hm.index, fontsize=14)

# 별표 — 시그니처 이름 바로 밑
blend = blended_transform_factory(ax_c.transAxes, ax_c.transData)
for si, sig_name in enumerate(hm.index):
    s = star_str(sig_pvals.get(sig_name, np.nan))
    if not s: continue
    col_s = '#c0392b' if s != 'ns' else '#888'
    ax_c.text(-0.01, si + 0.30, s,
              transform=blend, va='top', ha='right',
              fontsize=12, color=col_s, fontweight='bold')

cbar = plt.colorbar(im, ax=ax_c, shrink=0.75, pad=0.04)
cbar.ax.tick_params(labelsize=13)
cbar.set_label('z-score', fontsize=15)

ax_c.set_title('Transcriptional Signature Heatmap\n(z-score across 4 groups)',
               loc='center', fontsize=TIS, fontweight='bold', pad=TITLE_PAD)
add_panel_label(ax_c, 'C')

# ════════════════════════════════════════════════════════════════
# ── D: Violin (이전 E, 범례 우상단) ─────────────────────────────
# ════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[1, 1])
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
        vp = ax_d.violinplot([vals], positions=[xpos], widths=0.8,
                              showmedians=True, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(G_COLORS[grp])
            body.set_alpha(0.80 if grp=='PC6-High / SMA-Low' else 0.42)
            body.set_edgecolor('white')
        vp['cmedians'].set_color(G_COLORS[grp]); vp['cmedians'].set_linewidth(2.4)
    groups_vals = [rna_log.loc[gene,
                   [p for p in merged[merged['group4']==g]['pid'] if p in rna_log.columns]
                   ].values for g in G_ORDER]
    _, kp = kruskal(*groups_vals)
    sig = '***' if kp<0.001 else '**' if kp<0.01 else '*' if kp<0.05 else 'ns'
    ymax = max(v.max() for v in groups_vals)
    ax_d.text(group_x_base[mi]+1.5, ymax+0.15, sig,
              ha='center', fontsize=16, fontweight='bold',
              color='#c0392b' if sig!='ns' else '#888')
    ax_d.text(group_x_base[mi]+1.5, -0.7,
              f'{gene}\n({cat})', ha='center', fontsize=14, fontweight='bold', color=col)
ax_d.set_ylabel("log₂(TPM+1)", fontsize=LS)
ax_d.tick_params(axis='y', labelsize=TS)
ax_d.set_title('Representative Marker Expression\nby PC6 × SMA 4 Groups',
               loc='center', fontsize=TIS, fontweight='bold', pad=TITLE_PAD)
add_panel_label(ax_d, 'D')

ax_d.set_xlim(-0.8, group_x_base[-1]+3.8)
ax_d.set_xticks([])
ax_d.spines[['top','right']].set_visible(False)
handles_d = [mpatches.Patch(color=G_COLORS[g], label=g, alpha=0.75) for g in G_ORDER]
ax_d.legend(handles=handles_d, fontsize=13, loc='upper right',
            ncol=2).get_frame().set_edgecolor('#ccc')

# ╔══════════════════════════════════════════════════════════════╗
# ║                    Figure 제목 & 저장                          ║
# ╚══════════════════════════════════════════════════════════════╝

fig.suptitle('Joint Effects of PC6 and Body Composition',
             fontsize=23, fontweight='bold', y=0.945)

plt.savefig(OUT_TIFF, dpi=300, format='tiff', bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n저장 완료: {OUT_TIFF}")