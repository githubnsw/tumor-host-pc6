"""Supplementary Fig 6, FINAL v2, in the SUBMITTED five-panel layout (A, B gene-level bars; C pathway-level mean r; D TCGA gene-level
distribution with proliferation as negative control; E ssGSEA vs PC6). The notebook cell (Untitled3 cell 5) only draws A-B; its data
loading and gene-level Spearman block is reused verbatim (sandbox copy of FINAL v2 inputs), panels C-E are drawn from the same
gene-level table and from final_v2/outputs/results_ssgsea_pc6.csv (v2 ssGSEA, same five marker sets). Notebook untouched."""
import os, re, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); R1 = os.path.dirname(os.path.dirname(HERE))
S = r"C:\Users\nsw\AppData\Local\Temp\claude\c--Users-nsw-Desktop-body-composition\61cfca76-a232-4ce7-a356-6d92f663b762\scratchpad"
SB = os.path.join(S, "repro_final_v2").replace("\\", "/"); COX = r"C:\Users\nsw\anaconda3\envs\cox_env\python.exe"
src = open(os.path.join(S, "nb", "Untitled3_cell5.py"), encoding="utf-8").read()
for old in [r"C:\Users\nsw\Desktop\body_composition", r"C:\Users\nsw\body_com_paper_code"]: src = src.replace(old, SB)
cut = src.index("fig = plt.figure(figsize=(20, 11))"); head = src[:cut]
SSG = os.path.join(R1, "final_v2", "outputs", "results_ssgsea_pc6.csv").replace("\\", "/")
OUT = os.path.join(HERE, "SuppFig6_v2_fivepanel").replace("\\", "/")
TAIL = r'''
# ───────────────────────── five-panel figure (submitted layout) ─────────────────────────
corr_t.to_csv(r"%(OUT)s__gene_level_TCGA.csv"); corr_c.to_csv(r"%(OUT)s__gene_level_CPTAC.csv")
ssg = pd.read_csv(r"%(SSG)s")
def star(q): return '***' if q < 0.001 else '**' if q < 0.01 else '*' if q < 0.05 else 'ns'
SETS = list(GENE_SETS.keys())
order = []   # bottom-to-top order as submitted: glycolysis first (bottom) ... proliferation (top)
for s in SETS:
    for g in GENE_SETS[s]:
        if (s, g) not in order: order.append((s, g))
fig = plt.figure(figsize=(20, 14)); fig.patch.set_facecolor('white')
gs = GridSpec(3, 2, figure=fig, height_ratios=[1.35, 1.0, 0.95], hspace=0.45, wspace=0.25, left=0.07, right=0.98, top=0.93, bottom=0.05)
def gene_panel(ax, corr, title, letter):
    ys, labels, vals, cols, qs = [], [], [], [], []
    for i, (s, g) in enumerate(order):
        if g not in corr.index: continue
        ys.append(len(ys)); labels.append(g); vals.append(corr.loc[g, 'r']); cols.append(SET_COLORS[s]); qs.append(corr.loc[g, 'q'])
    ax.barh(ys, vals, color=cols, height=0.7)
    for y, v, q in zip(ys, vals, qs):
        if q < 0.05: ax.text(v + (0.012 if v >= 0 else -0.012), y, star(q), va='center', ha='left' if v >= 0 else 'right', fontsize=7.5, color='#C0392B')
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8); ax.axvline(0, color='#222', lw=0.9); ax.set_xlim(-0.65, 0.85)
    ax.set_xlabel('Spearman r  (vs PC6)', fontsize=11); ax.set_title(title, fontsize=12.5, fontweight='bold'); ax.tick_params(axis='x', labelsize=9)
    ax.text(-0.09, 1.02, letter, transform=ax.transAxes, fontsize=16, fontweight='bold')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SET_COLORS[s]) for s in SETS], labels=SETS, fontsize=8, loc='lower right', frameon=True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
gene_panel(fig.add_subplot(gs[0, 0]), corr_t, f'TCGA: Gene-level correlation with PC6 (n = {len(pc6_t)})', 'A')
gene_panel(fig.add_subplot(gs[0, 1]), corr_c, f'CPTAC-3: Gene-level correlation with PC6 (n = {len(pc6_c)})', 'B')
# C: pathway-level mean r
ax = fig.add_subplot(gs[1, 0]); x = np.arange(len(SETS)); w = 0.36; mt, mc = [], []
for s in SETS:
    gt = [g for g in GENE_SETS[s] if g in corr_t.index]; gc = [g for g in GENE_SETS[s] if g in corr_c.index]
    mt.append(corr_t.loc[gt, 'r'].mean()); mc.append(corr_c.loc[gc, 'r'].mean())
ax.bar(x - w / 2, mt, w, color=[SET_COLORS[s] for s in SETS], label='TCGA'); ax.bar(x + w / 2, mc, w, color=[SET_COLORS[s] for s in SETS], alpha=0.55, hatch='//', label='CPTAC-3')
for xi, (a, b) in enumerate(zip(mt, mc)):
    ax.text(xi - w / 2, a + (0.012 if a >= 0 else -0.03), f'{a:.2f}', ha='center', fontsize=8.5, fontweight='bold'); ax.text(xi + w / 2, b + (0.012 if b >= 0 else -0.03), f'{b:.2f}', ha='center', fontsize=8.5, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(SETS, fontsize=9.5, rotation=15); ax.axhline(0, color='#222', lw=0.9); ax.set_ylabel('Mean Spearman r  (vs PC6)', fontsize=10.5)
ax.set_title('Pathway-level mean r (TCGA vs CPTAC-3)', fontsize=12.5, fontweight='bold'); ax.text(-0.09, 1.02, 'C', transform=ax.transAxes, fontsize=16, fontweight='bold')
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='#E74C3C'), plt.Rectangle((0, 0), 1, 1, color='#E74C3C', alpha=0.55, hatch='//')], labels=['TCGA', 'CPTAC-3'], fontsize=9, loc='upper right', frameon=True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
print('PANEL C mean r TCGA:', dict(zip(SETS, np.round(mt, 3))), ' CPTAC:', dict(zip(SETS, np.round(mc, 3))))
# D: TCGA gene-level distribution, Warburg vs proliferation vs OXPHOS
ax = fig.add_subplot(gs[1, 1]); cats = [('OXPHOS', ['OXPHOS'], '#2980B9'), ('Proliferation\n(E2F / G2M / MKI67)', ['Proliferation'], '#7F8C8D'), ('Warburg\n(Glycolysis + Hypoxia + MCT)', ['Glycolysis', 'Hypoxia', 'Lactate/MCT'], '#E74C3C')]
for yi, (name, sets, col) in enumerate(cats):
    genes = sorted({g for s in sets for g in GENE_SETS[s] if g in corr_t.index}); vals = corr_t.loc[genes, 'r'].values
    jit = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.35
    ax.scatter(vals, yi + jit, color=col, s=28, alpha=0.85, zorder=3); m = vals.mean(); ax.plot([m, m], [yi - 0.28, yi + 0.28], color=col, lw=2.2, zorder=4)
    ax.text(m, yi - 0.42, f'mean r = {m:.2f}', ha='center', fontsize=9, color=col, fontweight='bold')
    for g, v, j in zip(genes, vals, jit): ax.text(v, yi + j + 0.06, g, fontsize=6, color=col, ha='center', alpha=0.9)
ax.set_yticks(range(len(cats))); ax.set_yticklabels([c[0] for c in cats], fontsize=9.5); ax.axvline(0, color='#999', lw=0.8, ls='--'); ax.set_xlim(-0.65, 0.85); ax.set_ylim(-0.7, 2.6)
ax.set_xlabel('Spearman r  (vs PC6, TCGA)', fontsize=10.5); ax.set_title('PC6 ~ Warburg vs Proliferation vs OXPHOS\n(TCGA, proliferation as negative control)', fontsize=12, fontweight='bold')
ax.text(-0.22, 1.02, 'D', transform=ax.transAxes, fontsize=16, fontweight='bold')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
# E: ssGSEA vs PC6 (FINAL v2)
ax = fig.add_subplot(gs[2, :]); x = np.arange(len(SETS)); w = 0.3
for k, (coh, off, alpha, hatch) in enumerate([('TCGA', -w / 2, 1.0, None), ('CPTAC-3', w / 2, 0.55, '//')]):
    for xi, s in enumerate(SETS):
        row = ssg[(ssg.cohort == coh) & (ssg.pathway == s)].iloc[0]; r, q = row.r, row.q
        ax.bar(xi + off, r, w, color=SET_COLORS[s], alpha=alpha, hatch=hatch)
        ax.text(xi + off, r / 2 if abs(r) > 0.12 else (r + 0.03 if r >= 0 else r - 0.05), f'{r:+.2f}', ha='center', va='center', fontsize=9, color='white' if abs(r) > 0.12 else '#333', fontweight='bold')
        ax.text(xi + off, (r + 0.05) if r >= 0 else 0.04, star(q), ha='center', fontsize=9.5, color='#C0392B' if q < 0.05 else '#777', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(SETS, fontsize=10.5); ax.axhline(0, color='#222', lw=0.9); ax.set_ylim(-0.25, 0.85); ax.set_ylabel('Spearman r  (ssGSEA NES vs PC6)', fontsize=10.5)
ax.set_title('ssGSEA Pathway Enrichment Score ~ PC6  (Panel E)\nSingle-sample GSEA, rank normalization | FDR-corrected (Benjamini-Hochberg)', fontsize=12, fontweight='bold')
ax.text(-0.04, 1.02, 'E', transform=ax.transAxes, fontsize=16, fontweight='bold')
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='#E74C3C'), plt.Rectangle((0, 0), 1, 1, color='#E74C3C', alpha=0.55, hatch='//')], labels=['TCGA', 'CPTAC-3'], fontsize=9.5, loc='upper right', frameon=True)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
fig.suptitle('PC6 Represents a Conserved Warburg-like Metabolic Program', fontsize=15, fontweight='bold', y=0.975)
plt.savefig(r"%(OUT)s.tiff", dpi=300, format='tiff', bbox_inches='tight', facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
plt.savefig(r"%(OUT)s.png", dpi=110, bbox_inches='tight', facecolor='white'); plt.close()
print('saved', r"%(OUT)s.tiff")
''' % dict(OUT=OUT, SSG=SSG)
script = os.path.join(HERE, "run_Untitled3_cell5_fivepanel.py"); open(script, "w", encoding="utf-8").write("import matplotlib\nmatplotlib.use('Agg')\n" + head + TAIL)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=os.path.join(S, "pylib_cox"), MPLBACKEND="Agg"); t0 = time.time()
with open(os.path.join(HERE, "run.log"), "w", encoding="utf-8") as lf:
    rc = subprocess.run([COX, "-s", script], cwd=SB, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
log = open(os.path.join(HERE, "run.log"), encoding="utf-8").read(); print(f"rc={rc} {time.time()-t0:.0f}s"); print(log[-2500:])
