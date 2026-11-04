"""
Figure 2. Autoencoder Latent Program Discovery and Dynamic PC Selection
======================================================================
Panel A: Forest plot  — HR ± 95%CI, TCGA + External
Panel B: -log10(p) scatter — bilateral significance
Panel C: Dynamic selection workflow

입력:  TCGA_with_genes.csv, External_with_genes.csv
       ae_latent_pca10_all.csv, ae_latent_pca10_external.csv
출력:  Figure2.tiff  (dpi=300)
실행:  python figure2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from lifelines import CoxPHFitter
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
OUTPATH   = os.path.join(BASE, "Figure2.tiff")


# ============================================================
# Helpers
# ============================================================
def star(p):
    if pd.isna(p):
        return 'na'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def safe_neglog10(p):
    if pd.isna(p):
        return 0.0
    return -np.log10(max(float(p), 1e-300))


def same_direction(hr1, hr2):
    if pd.isna(hr1) or pd.isna(hr2):
        return False
    return ((hr1 > 1) and (hr2 > 1)) or ((hr1 < 1) and (hr2 < 1))


def fmt_hr(hr, lo, hi, p):
    if any(pd.isna(x) for x in [hr, lo, hi, p]):
        return "NA"
    return f"HR={hr:.3f} ({lo:.3f}–{hi:.3f})  {star(p)}"


def choose_selected_pc(screen_df):
    df = screen_df.copy()

    df['t_sig'] = df['t_p'] < 0.05
    df['e_sig'] = df['e_p'] < 0.05
    df['same_dir'] = [
        same_direction(t_hr, e_hr)
        for t_hr, e_hr in zip(df['t_hr'], df['e_hr'])
    ]
    df['bilateral'] = df['t_sig'] & df['e_sig'] & df['same_dir']
    df['score'] = df['t_p'].apply(safe_neglog10) + df['e_p'].apply(safe_neglog10)
    df['effect_score'] = np.abs(df['t_hr'] - 1) + np.abs(df['e_hr'] - 1)

    bilateral_df = df[df['bilateral']].copy()
    if len(bilateral_df) == 0:
        return None, df

    bilateral_df = bilateral_df.sort_values(
        by=['score', 'effect_score'],
        ascending=[False, False]
    ).reset_index(drop=True)

    selected_pc = bilateral_df.loc[0, 'PC']
    return selected_pc, df


def make_step1_pass_text(df):
    passed = df[df['t_sig']]['PC'].tolist()
    if not passed:
        return "Pass: none (0 of 10)"
    pass_txt = ", ".join(
        [f"{pc} {star(df.loc[df['PC'] == pc, 't_p'].iloc[0])}" for pc in passed]
    )
    return f"Pass: {pass_txt}\n({len(passed)} of 10)"


def make_step2_result_lines(df, tcga_passed_pcs, selected_pc):
    lines = []
    for pc in tcga_passed_pcs:
        row = df[df['PC'] == pc].iloc[0]
        ok = row['e_sig'] and row['same_dir']
        mark = "✓" if ok else "✗"
        line = f"{pc}: p={row['e_p']:.3f} {mark}"
        lines.append((pc, line, ok))
    return lines


def make_eliminated_text(df):
    tcga_ns = df.loc[~df['t_sig'], 'PC'].tolist()
    ext_fail = df.loc[df['t_sig'] & ~(df['e_sig'] & df['same_dir']), 'PC'].tolist()

    tcga_ns_txt = ", ".join(tcga_ns) if tcga_ns else "none"
    ext_fail_txt = ", ".join(ext_fail) if ext_fail else "none"

    return (
        f"Eliminated: {tcga_ns_txt}  (TCGA ns)",
        f"Replication failed: {ext_fail_txt}"
    )


# ============================================================
# Load & prep
# ============================================================
print("Loading data...")
t = pd.read_csv(TCGA_PATH)
e = pd.read_csv(EXT_PATH)
ae_t = pd.read_csv(AE_TCGA)
ae_e = pd.read_csv(AE_EXT)

for df in [t, e]:
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['OS_time'] = pd.to_numeric(df['OS_time'], errors='coerce') / 30.44
    df['sex_bin'] = (df['sex'].astype(str).str.upper() == 'MALE').astype(float)
    df['stage_bin'] = df['stage'].apply(
        lambda s: 1.0 if any(x in str(s) for x in ['III', 'IV'])
        else (0.0 if any(x in str(s) for x in ['I', 'II']) else np.nan)
    )

t['pid'] = t['patient_id'].astype(str).str[:12]
ae_t['pid'] = ae_t['patient_id'].astype(str).str[:12]
e['pid'] = e['patient_id'].astype(str)
ae_e['pid'] = ae_e['patient_id'].astype(str)

for df_clin, ae_df in [(t, ae_t), (e, ae_e)]:
    merged = df_clin.merge(
        ae_df[['pid'] + [f'AE_PC{i}' for i in range(1, 11)]],
        on='pid',
        how='inner'
    )
    for i in range(1, 11):
        col = f'AE_PC{i}'
        std = merged[col].std()
        if pd.isna(std) or std == 0:
            merged[f'{col}_z'] = np.nan
        else:
            merged[f'{col}_z'] = (merged[col] - merged[col].mean()) / std

    if df_clin is t:
        df_t = merged
    else:
        df_e = merged

print(f"  TCGA: {len(df_t)}  |  External: {len(df_e)}")


# ============================================================
# Multivariable Cox: each PC (adj SMA+age+sex+stage, strat cancer)
# ============================================================
print("Computing Cox...")
rows = []

for i in range(1, 11):
    pc_col = f'AE_PC{i}_z'
    row = {'PC': f'PC{i}'}

    for tag, df_ in [('t', df_t), ('e', df_e)]:
        d = df_[[
            'OS_time', 'OS', 'cancer_type', 'age', 'sex_bin',
            'stage_bin', 'SMA_zscore', pc_col
        ]].dropna()

        d = d[d['OS_time'] > 0]

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(d, duration_col='OS_time', event_col='OS', strata='cancer_type')
            r = cph.summary.loc[pc_col]

            row[f'{tag}_hr'] = r['exp(coef)']
            row[f'{tag}_lo'] = r['exp(coef) lower 95%']
            row[f'{tag}_hi'] = r['exp(coef) upper 95%']
            row[f'{tag}_p'] = r['p']
        except Exception:
            row[f'{tag}_hr'] = np.nan
            row[f'{tag}_lo'] = np.nan
            row[f'{tag}_hi'] = np.nan
            row[f'{tag}_p'] = 1.0

    print(
        f"  PC{i}: "
        f"TCGA HR={row['t_hr']:.3f} p={row['t_p']:.4f} | "
        f"Ext HR={row['e_hr']:.3f} p={row['e_p']:.4f}"
    )
    rows.append(row)

SCREEN = pd.DataFrame(rows)

selected_pc, SCREEN = choose_selected_pc(SCREEN)
selected_exists = selected_pc is not None
selected_idx = None
if selected_exists:
    selected_idx = SCREEN.index[SCREEN['PC'] == selected_pc][0]

SIG = -np.log10(0.05)

tcga_passed_pcs = SCREEN.loc[SCREEN['t_sig'], 'PC'].tolist()
step1_pass_text = make_step1_pass_text(SCREEN)
step2_lines = make_step2_result_lines(SCREEN, tcga_passed_pcs, selected_pc)
elim_line1, elim_line2 = make_eliminated_text(SCREEN)

if selected_exists:
    sel_row = SCREEN[SCREEN['PC'] == selected_pc].iloc[0]
    direction_text = "same direction" if sel_row['same_dir'] else "discordant direction"
else:
    sel_row = None
    direction_text = "no replicated signal"


# ============================================================
# Figure  (1 row x 3 panels)
# ============================================================
fig = plt.figure(figsize=(30, 10))
fig.patch.set_facecolor('white')
gs = GridSpec(
    1, 3, figure=fig,
    wspace=0.42,
    left=0.05, right=0.97,
    top=0.88, bottom=0.10,
    width_ratios=[1.2, 0.90, 0.85]
)


# ──────────────────────────────────────────────────────────────
# Panel A: Forest plot
# ──────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

n = len(SCREEN)
ys = np.arange(n)[::-1]
OFF = 0.20

ax_a.axvline(1, color='#888', lw=1.2, ls='--', zorder=0)

for i, row in SCREEN.iterrows():
    y = ys[i]
    is_selected = selected_exists and (row['PC'] == selected_pc)
    t_s = row['t_sig']
    e_s = row['e_sig']

    if is_selected:
        ax_a.axhspan(y - 0.46, y + 0.46, color='#FADBD8', alpha=0.45, zorder=1)
    elif i % 2 == 0:
        ax_a.axhspan(y - 0.46, y + 0.46, color='#F8F9FA', alpha=0.4, zorder=0)

    tc = '#2980B9' if t_s else '#AED6F1'
    ax_a.errorbar(
        row['t_hr'], y + OFF,
        xerr=[[row['t_hr'] - row['t_lo']], [row['t_hi'] - row['t_hr']]],
        fmt='D' if t_s else 'o',
        color=tc,
        markersize=11 if (t_s and is_selected) else (9 if t_s else 5),
        elinewidth=2 if t_s else 1.0,
        capsize=4,
        zorder=4,
        markeredgecolor='#111' if t_s else tc,
        markeredgewidth=0.9 if t_s else 0
    )

    ec = '#E74C3C' if e_s else '#F1948A'
    ax_a.errorbar(
        row['e_hr'], y - OFF,
        xerr=[[row['e_hr'] - row['e_lo']], [row['e_hi'] - row['e_hr']]],
        fmt='s' if e_s else 'o',
        color=ec,
        markersize=11 if (e_s and is_selected) else (9 if e_s else 5),
        elinewidth=2 if e_s else 1.0,
        capsize=4,
        zorder=4,
        markeredgecolor='#111' if e_s else ec,
        markeredgewidth=0.9 if e_s else 0
    )

    ax_a.text(
        -0.02, y, row['PC'],
        va='center', ha='right', fontsize=11,
        fontweight='bold' if is_selected else 'normal',
        color='#C0392B' if is_selected else '#2C3E50',
        transform=ax_a.get_yaxis_transform()
    )

    ax_a.text(
        2.90, y + OFF,
        fmt_hr(row['t_hr'], row['t_lo'], row['t_hi'], row['t_p']),
        va='center', ha='left', fontsize=8.5,
        color='#2980B9' if t_s else '#CCC',
        fontweight='bold' if t_s else 'normal'
    )
    ax_a.text(
        2.90, y - OFF,
        fmt_hr(row['e_hr'], row['e_lo'], row['e_hi'], row['e_p']),
        va='center', ha='left', fontsize=8.5,
        color='#E74C3C' if e_s else '#CCC',
        fontweight='bold' if e_s else 'normal'
    )

    if is_selected:
        for yo, co in [(OFF, '#2980B9'), (-OFF, '#E74C3C')]:
            ax_a.text(
                0.03, y + yo, '★',
                va='center', ha='left', fontsize=11, color=co, zorder=5,
                transform=ax_a.get_yaxis_transform()
            )

ax_a.set_xlim(0.45, 2.85)
ax_a.set_ylim(-0.8, n - 0.15)
ax_a.set_yticks([])
ax_a.set_xlabel('Hazard Ratio  (95% CI)', fontsize=12)
ax_a.set_title('A', fontsize=14, fontweight='bold', loc='left')
ax_a.set_title(
    'PC1–PC10 Multivariable Cox Forest Plot\n'
    'adj: SMA, age, sex, stage  |  strat: cancer type  |  OS',
    fontsize=11.5, fontweight='bold', loc='center'
)
ax_a.text(
    2.88, n - 0.42, 'HR (95% CI)   sig',
    va='center', ha='left', fontsize=9, fontweight='bold', color='#555'
)

leg_a = [
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#2980B9',
           markeredgecolor='#111', markersize=10, label='TCGA  p<0.05'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#E74C3C',
           markeredgecolor='#111', markersize=10, label='External  p<0.05'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#BDC3C7',
           markersize=7, label='ns'),
]

if selected_exists:
    leg_a.append(
        mpatches.Patch(
            facecolor='#FADBD8', alpha=0.6, edgecolor='#C0392B',
            label=f'{selected_pc}  (selected)'
        )
    )

ax_a.legend(handles=leg_a, fontsize=9, framealpha=0.9, loc='lower right')
for sp in ['top', 'right', 'left']:
    ax_a.spines[sp].set_visible(False)


# ──────────────────────────────────────────────────────────────
# Panel B: -log10(p) scatter
# ──────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

t_lp = [safe_neglog10(p) for p in SCREEN['t_p']]
e_lp = [safe_neglog10(p) for p in SCREEN['e_p']]
eff = np.abs(SCREEN['t_hr'].values - 1) + np.abs(SCREEN['e_hr'].values - 1)
eff = np.nan_to_num(eff, nan=0.0)
sz = eff * 700 + 60

xmax = max(max(t_lp) * 1.5, SIG + 0.8, 2.0)
ymax = max(max(e_lp) * 1.9, SIG + 0.8, 2.0)

ax_b.fill_between([SIG, xmax], SIG, ymax, color='#FADBD8', alpha=0.25, zorder=0)
ax_b.axvline(SIG, color='#2980B9', lw=1.2, ls='--', alpha=0.65)
ax_b.axhline(SIG, color='#E74C3C', lw=1.2, ls='--', alpha=0.65)
ax_b.text(SIG + 0.04, 0.05, 'TCGA p=0.05',
          color='#2980B9', fontsize=8.5, rotation=90, va='bottom')
ax_b.text(0.05, SIG + 0.04, 'External p=0.05',
          color='#E74C3C', fontsize=8.5, va='bottom')

for i, row in SCREEN.iterrows():
    is_selected = selected_exists and (row['PC'] == selected_pc)
    t_s = row['t_sig']
    e_s = row['e_sig']
    bilateral = row['bilateral']

    if is_selected:
        col = '#C0392B'
    elif bilateral:
        col = '#27AE60'
    elif t_s and not e_s:
        col = '#2980B9'
    elif e_s and not t_s:
        col = '#9B59B6'
    else:
        col = '#BDC3C7'

    ax_b.scatter(
        t_lp[i], e_lp[i],
        s=sz[i],
        c=col,
        zorder=4 if is_selected else 3,
        edgecolors='#111' if is_selected else col,
        linewidths=1.8 if is_selected else 0.4,
        alpha=0.90
    )

    ox, oy = 0.06, 0.06
    ax_b.annotate(
        row['PC'],
        xy=(t_lp[i], e_lp[i]),
        xytext=(t_lp[i] + ox, e_lp[i] + oy),
        fontsize=10.5 if is_selected else 9.5,
        fontweight='bold' if is_selected else 'normal',
        color='#C0392B' if is_selected else '#444'
    )

if selected_exists:
    ax_b.annotate(
        (
            f'★  {selected_pc}\n'
            f'TCGA p={sel_row["t_p"]:.3f} {star(sel_row["t_p"])}\n'
            f'External p={sel_row["e_p"]:.3f} {star(sel_row["e_p"])}\n'
            f'→ Bilateral + {direction_text}'
        ),
        xy=(t_lp[selected_idx], e_lp[selected_idx]),
        xytext=(t_lp[selected_idx] + 0.40, e_lp[selected_idx] + 0.55),
        fontsize=9.5,
        color='#C0392B',
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.30',
            facecolor='#FADBD8',
            edgecolor='#C0392B',
            alpha=0.92
        ),
        arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.8)
    )
else:
    ax_b.text(
        0.97, 0.92,
        'No PC passed bilateral replication\n(TCGA p<0.05 + External p<0.05 + same direction)',
        transform=ax_b.transAxes,
        ha='right', va='top',
        fontsize=9.2,
        color='#7B7D7D',
        bbox=dict(boxstyle='round,pad=0.30', facecolor='#F8F9F9', edgecolor='#CCD1D1')
    )

ax_b.set_xlabel('−log₁₀(p)  TCGA', fontsize=12)
ax_b.set_ylabel('−log₁₀(p)  External', fontsize=12)
ax_b.set_xlim(0, xmax)
ax_b.set_ylim(0, ymax)
ax_b.set_title('B', fontsize=14, fontweight='bold', loc='left')
ax_b.set_title('Bilateral Significance\nTCGA vs External',
               fontsize=11.5, fontweight='bold', loc='center')
ax_b.text(
    0.97, 0.03,
    'Shaded: both significant\nBubble size ∝ effect size',
    transform=ax_b.transAxes,
    ha='right', va='bottom',
    fontsize=8.5, style='italic', color='#555'
)
for sp in ['top', 'right']:
    ax_b.spines[sp].set_visible(False)


# ──────────────────────────────────────────────────────────────
# Panel C: Flowchart
# ──────────────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_xlim(0, 10)
ax_c.set_ylim(0, 12)
ax_c.axis('off')
ax_c.set_title('C', fontsize=14, fontweight='bold', loc='left')
ax_c.set_title('Dynamic PC Selection Workflow',
               fontsize=11.5, fontweight='bold', loc='center')


def fbox(ax, x, y, w, h, title, body, fc, ec, tc='white'):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.12',
            facecolor=fc, edgecolor=ec, linewidth=2, zorder=3
        )
    )
    ax.text(
        x + w / 2, y + h - 0.26, title,
        ha='center', va='top',
        fontsize=9.8, fontweight='bold', color=tc, zorder=4
    )
    ax.text(
        x + w / 2, y + h / 2 - 0.15, body,
        ha='center', va='center',
        fontsize=8.5, color='#2C3E50', zorder=4, linespacing=1.4
    )


def arr(ax, x, y0, y1):
    ax.annotate(
        '', xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle='->,head_width=0.25', color='#555', lw=2)
    )


fbox(
    ax_c, 1.5, 10.5, 7.0, 1.0,
    'Input: AE Latent Space',
    'PC1–PC10  (10 components per patient)',
    '#2C3E50', '#1A252F'
)
arr(ax_c, 5.0, 10.5, 9.8)

fbox(
    ax_c, 1.5, 8.5, 7.0, 1.2,
    'Step 1.  TCGA Multivariable Cox OS',
    'adj: SMA, age, sex, stage  |  strat: cancer type\nSignificance: p < 0.05',
    '#EBF5FB', '#2980B9', '#2980B9'
)
arr(ax_c, 5.0, 8.5, 7.7)
ax_c.text(
    8.7, 9.1, step1_pass_text,
    ha='left', va='center',
    fontsize=8, color='#27AE60', fontweight='bold'
)

fbox(
    ax_c, 1.5, 6.5, 7.0, 1.2,
    'Step 2.  External Validation',
    f'n={len(df_e)}  |  same direction + p < 0.05',
    '#EBF5FB', '#E74C3C', '#E74C3C'
)
arr(ax_c, 5.0, 6.5, 5.7)

y_positions = [7.15, 6.87, 6.59, 6.31, 6.03]
for j, (_, line, ok) in enumerate(step2_lines[:5]):
    ax_c.text(
        8.7, y_positions[j], line,
        ha='left', va='center',
        fontsize=8.3 if ok else 8.0,
        color='#C0392B' if ok else '#E74C3C',
        fontweight='bold' if ok else 'normal'
    )

fbox(
    ax_c, 1.5, 4.5, 7.0, 1.2,
    'Step 3.  Selection Rule',
    'Choose PC with bilateral replication\nmaximize: -log10(TCGA p) + -log10(External p)',
    '#FEF9E7', '#E67E22', '#E67E22'
)
arr(ax_c, 5.0, 4.5, 3.7)

if selected_exists:
    ax_c.add_patch(
        mpatches.FancyBboxPatch(
            (1.5, 2.5), 7.0, 1.4,
            boxstyle='round,pad=0.12',
            facecolor='#C0392B',
            edgecolor='#922B21',
            linewidth=2.5,
            zorder=3
        )
    )
    ax_c.text(
        5.0, 3.55, f'★   {selected_pc}   SELECTED',
        ha='center', va='center',
        fontsize=13, fontweight='bold', color='white', zorder=4
    )
    ax_c.text(
        5.0, 2.98,
        (
            f'TCGA  HR={sel_row["t_hr"]:.3f} ({sel_row["t_lo"]:.3f}–{sel_row["t_hi"]:.3f})  p={sel_row["t_p"]:.3f}\n'
            f'External  HR={sel_row["e_hr"]:.3f} ({sel_row["e_lo"]:.3f}–{sel_row["e_hi"]:.3f})  p={sel_row["e_p"]:.3f}'
        ),
        ha='center', va='center',
        fontsize=8.8, color='#FADBD8', zorder=4, linespacing=1.4
    )
    bottom_lbl = f'{selected_pc} ★'
else:
    ax_c.add_patch(
        mpatches.FancyBboxPatch(
            (1.5, 2.5), 7.0, 1.4,
            boxstyle='round,pad=0.12',
            facecolor='#7F8C8D',
            edgecolor='#566573',
            linewidth=2.5,
            zorder=3
        )
    )
    ax_c.text(
        5.0, 3.55, 'NO PC SELECTED',
        ha='center', va='center',
        fontsize=13, fontweight='bold', color='white', zorder=4
    )
    ax_c.text(
        5.0, 2.98,
        'No component satisfied\nbilateral significance + same direction',
        ha='center', va='center',
        fontsize=8.8, color='#ECF0F1', zorder=4, linespacing=1.4
    )
    bottom_lbl = '0 selected'

ax_c.add_patch(
    mpatches.FancyBboxPatch(
        (1.5, 1.1), 7.0, 1.05,
        boxstyle='round,pad=0.10',
        facecolor='#F2F3F4',
        edgecolor='#CCC',
        linewidth=1.2,
        zorder=2
    )
)
ax_c.text(
    5.0, 1.90, elim_line1,
    ha='center', va='center',
    fontsize=8.8, fontweight='bold', color='#888', zorder=3
)
ax_c.text(
    5.0, 1.42, elim_line2,
    ha='center', va='center',
    fontsize=8.2, color='#AAA', zorder=3
)

flow_labels = [
    (9.75, '10 PCs'),
    (8.00, f'{len(tcga_passed_pcs)} PCs'),
    (6.00, f'{int(SCREEN["bilateral"].sum())} PCs'),
    (3.20, bottom_lbl)
]
for yp, lbl in flow_labels:
    is_bottom = (lbl == bottom_lbl and selected_exists)
    ax_c.text(
        0.55, yp, lbl,
        ha='center', va='center', fontsize=8.5,
        fontweight='bold' if is_bottom else 'normal',
        color='#C0392B' if is_bottom else '#555',
        bbox=dict(
            boxstyle='round,pad=0.2',
            facecolor='#FADBD8' if is_bottom else 'white',
            edgecolor='#C0392B' if is_bottom else '#CCC',
            linewidth=1.2
        )
    )


# ============================================================
# Figure title
# ============================================================
if selected_exists:
    fig_title = (
        f'Figure 2.  Autoencoder Latent Program Discovery and {selected_pc} Selection\n'
        f'{selected_pc} showed bilateral prognostic replication with concordant effect direction.'
    )
else:
    fig_title = (
        'Figure 2.  Autoencoder Latent Program Discovery and Dynamic PC Screening\n'
        'No component satisfied bilateral prognostic replication with concordant effect direction.'
    )

fig.suptitle(
    fig_title,
    fontsize=12.5, fontweight='bold', y=1.01
)

plt.savefig(
    OUTPATH,
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    format='tiff'
)
plt.close()

print(f"Selected PC: {selected_pc}")
print(f"Saved: {OUTPATH}")