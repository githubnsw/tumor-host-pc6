import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── 데이터 ──────────────────────────────────────────────
full_data = [
    ('BLCA',  105, 49,  1.03,  0.92, 1.15, 0.635),
    ('COAD',  21,  8,   1.03,  0.50, 2.10, 0.944),
    ('KIRC',  171, 47,  1.07,  0.92, 1.24, 0.385),
    ('KIRP',  22,  6,   0.95,  0.46, 1.94, 0.881),
    ('LIHC',  72,  34,  1.38,  1.18, 1.61, 0.000),
    ('STAD',  43,  28,  0.82,  0.62, 1.09, 0.173),
]
full_overall = ('Overall', 494, 192, 1.065, 1.002, 1.132, 0.042)

nokirc_data = [
    ('BLCA',  105, 49,  1.03,  0.92, 1.15, 0.635),
    ('COAD',  21,  8,   1.03,  0.50, 2.10, 0.944),
    ('KIRP',  22,  6,   0.95,  0.46, 1.94, 0.881),
    ('LIHC',  72,  34,  1.38,  1.18, 1.61, 0.000),
    ('STAD',  43,  28,  0.82,  0.62, 1.09, 0.173),
]
nokirc_overall = ('Overall', 320, 129, 1.068, 0.997, 1.143, 0.059)

XMIN, XMAX = 0.3, 2.5
X_REF = 1.0

def draw_panel(fig, gs_col, rows, overall, title):
    """
    각 패널을 3개 서브축으로 구성:
      ax_left  : Cancer Type | N (events)  텍스트
      ax_mid   : Forest plot (CI 선 + 포인트)
      ax_right : HR (95% CI), p-value 텍스트
    """
    n_rows = len(rows) + 2   # rows + 구분줄 gap + overall

    # 3-column grid: 왼쪽 텍스트 | 플롯 | 오른쪽 텍스트
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_col,
        width_ratios=[1.8, 3.0, 2.2], wspace=0
    )
    ax_l = fig.add_subplot(inner[0])
    ax_m = fig.add_subplot(inner[1])
    ax_r = fig.add_subplot(inner[2])

    for ax in [ax_l, ax_m, ax_r]:
        ax.set_ylim(-0.8, n_rows + 0.5)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # 중앙 플롯 축만 x축 표시
    ax_m.set_xlim(XMIN, XMAX)
    ax_m.spines['bottom'].set_visible(True)
    ax_m.set_xlabel('Hazard Ratio (PC6)', fontsize=9, color='#333333')
    ax_m.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax_m.tick_params(axis='x', labelsize=8.5)
    ax_m.axvline(X_REF, color='#777777', lw=1.0, linestyle='--', alpha=0.7, zorder=1)

    ax_l.set_xlim(0, 1)
    ax_r.set_xlim(0, 1)
    for ax in [ax_l, ax_r]:
        ax.tick_params(left=False, bottom=False, labelbottom=False)

    # 헤더
    ax_l.text(0.04, n_rows + 0.1, 'Cancer Type',
              fontsize=8.5, fontweight='bold', color='#333333', va='bottom')
    ax_l.text(0.72, n_rows + 0.1, 'N (events)',
              fontsize=8.5, fontweight='bold', color='#333333', va='bottom', ha='center')
    ax_r.text(0.5, n_rows + 0.1, 'HR (95% CI)      p-value',
              fontsize=8.5, fontweight='bold', color='#333333', va='bottom', ha='center')

    # 제목
    ax_m.set_title(title, fontsize=9.5, fontweight='bold',
                   color='#1A3A6B', pad=10)

    def plot_row(i, label, N, events, HR, low, high, p, is_overall=False):
        y = n_rows - 1 - i  # 위→아래

        # 배경
        shade = '#F0F4FA' if (i % 2 == 0 and not is_overall) else 'white'
        if not is_overall:
            for ax in [ax_l, ax_m, ax_r]:
                ax.axhspan(y - 0.42, y + 0.42, color=shade, zorder=0)

        color  = '#1A3A6B' if is_overall else '#2874A6'
        marker = 'D'        if is_overall else 's'
        ms     = 80         if is_overall else 40
        lw     = 2.2        if is_overall else 1.5
        fs     = 9.5        if is_overall else 9.0
        fw     = 'bold'     if is_overall else 'normal'

        # 왼쪽: 암종명
        ax_l.text(0.04, y, label,
                  va='center', ha='left', fontsize=fs, fontweight=fw,
                  color='#1A3A6B' if is_overall else '#222222')

        # 왼쪽: N (events) — 오른쪽 정렬
        ax_l.text(0.96, y, f'{N} ({events})',
                  va='center', ha='right', fontsize=8.2,
                  color='#555555', style='italic' if not is_overall else 'normal',
                  fontweight=fw)

        # 중앙: CI 선 + 포인트 (클리핑 처리)
        lo_clip = max(low,  XMIN + 0.02)
        hi_clip = min(high, XMAX - 0.02)
        ax_m.plot([lo_clip, hi_clip], [y, y],
                  color=color, lw=lw, zorder=2, solid_capstyle='round')
        ax_m.scatter(HR, y, color=color, s=ms, zorder=3,
                     marker=marker, linewidths=0)

        # 오른쪽: HR (CI), p
        p_str = 'p<0.001' if p < 0.001 else f'p={p:.3f}'
        sig   = '*' if p < 0.05 else ''
        hr_str = f'{HR:.2f} ({low:.2f}–{high:.2f})'

        txt_color = '#C0392B' if (is_overall and p < 0.05) else \
                    ('#1A3A6B' if is_overall else '#333333')

        ax_r.text(0.08, y, hr_str,
                  va='center', ha='left', fontsize=fs - 0.5, fontweight=fw,
                  color=txt_color)
        ax_r.text(0.75, y, f'{p_str}{sig}',
                  va='center', ha='left', fontsize=fs - 0.5, fontweight=fw,
                  color=txt_color)

    # cancer-type 행
    for i, row in enumerate(rows):
        plot_row(i, *row)

    # 구분선
    sep_y = n_rows - len(rows) - 1
    for ax in [ax_l, ax_m, ax_r]:
        ax.axhline(sep_y + 0.5, color='#AAAAAA', lw=0.9)

    # overall 행
    plot_row(len(rows) + 1, *overall, is_overall=True)
    for ax in [ax_l, ax_m, ax_r]:
        ax.axhspan(-0.7, sep_y + 0.3, color='#EEF3FA', zorder=0, alpha=0.5)


# ── Figure ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 6))
fig.patch.set_facecolor('white')

outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12,
                          left=0.01, right=0.99, top=0.88, bottom=0.12)

draw_panel(fig, outer[0], full_data,    full_overall,
           'Full Cohort  (N=494, Events=192)')
draw_panel(fig, outer[1], nokirc_data,  nokirc_overall,
           'KIRC + READ Excluded  (N=320, Events=129)')

fig.suptitle(
    'Sensitivity Analysis: PC6 Prognostic Effect\n'
    'Full Cohort vs. KIRC+READ Excluded',
    fontsize=12, fontweight='bold', color='#1A3A6B', y=0.98
)

plt.savefig(r'C:\Users\nsw\body_com_paper_code\supp_figure_sensitivity_forest_v2.tiff',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Done")