"""Supp Fig 2 (Untitled1_cell1) re-run on FINAL v2 with the UNIFIED spec in panel C (binary stage I/II vs III/IV, missing excluded;
already stratified by cancer type and unpenalised in the submitted cell) and a stratified log-rank p added under the pooled log-rank p
in panels A-B. Also fits, as a sensitivity, one 4-level Cox model (3 dummies vs IH_BG) instead of the submitted pairwise models.
Sandbox = scratchpad/repro_final_v2; notebook untouched. Outputs: Figure2_unified.tiff/png, supp_fig2_numbers.csv, run.log"""
import os, re, subprocess, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S = r"C:\Users\nsw\AppData\Local\Temp\claude\c--Users-nsw-Desktop-body-composition\61cfca76-a232-4ce7-a356-6d92f663b762\scratchpad"
SB = os.path.join(S, "repro_final_v2").replace("\\", "/"); COX = r"C:\Users\nsw\anaconda3\envs\cox_env\python.exe"
src = open(os.path.join(S, "nb", "Untitled1_cell1.py"), encoding="utf-8").read()
for old in [r"C:\Users\nsw\Desktop\body_composition", r"C:\Users\nsw\body_com_paper_code"]: src = src.replace(old, SB)
STRAT = '''
def _stratified_logrank(df, T='OS_time', E='OS', G='group', S_='cancer_type'):  # same implementation as r2_8 / fig4_final
    from scipy.stats import chi2 as _chi2
    d0 = df[[T, E, G, S_]].dropna(); d0 = d0[d0[T] > 0]
    groups = sorted(d0[G].unique()); k = len(groups); OE = np.zeros(k); V = np.zeros((k, k))
    for _, d in d0.groupby(S_):
        for t in np.sort(d.loc[d[E] == 1, T].unique()):
            at = d[d[T] >= t]; n = len(at); dead = d[(d[T] == t) & (d[E] == 1)]; dt = len(dead)
            if n < 2 or dt == 0: continue
            nj = np.array([(at[G] == g).sum() for g in groups], float); dj = np.array([(dead[G] == g).sum() for g in groups], float)
            p = nj / n; OE += dj - dt * p; V += dt * (n - dt) / (n - 1) * (np.diag(p) - np.outer(p, p))
    oe, v = OE[:-1], V[:-1, :-1]; stat = float(oe @ np.linalg.pinv(v) @ oe); return float(_chi2.sf(stat, k - 1))
tcga_p_strat = _stratified_logrank(tcga); cptac_p_strat = _stratified_logrank(cptac)
STRAT_P = {'A': tcga_p_strat, 'B': cptac_p_strat}
print(f"stratified log-rank: TCGA p={tcga_p_strat:.4g}, CPTAC p={cptac_p_strat:.4g}")
'''
P = [
    ("def clean_stage(s):\n    if pd.isna(s): return np.nan\n    s = str(s).upper()\n    if 'IV'  in s: return 4\n    if 'III' in s: return 3\n    if 'II'  in s: return 2\n    if 'I'   in s: return 1\n    return np.nan",
     "def clean_stage(s):  # unified spec: binary stage, missing/unknown -> NaN (excluded)\n    s = str(s).upper().replace('STAGE', '').strip()\n"
     "    if s in ('', 'NAN', 'NONE', 'UNKNOWN', '[DISCREPANCY]', '[NOT AVAILABLE]', '[NOT APPLICABLE]'): return np.nan\n"
     "    if s.startswith(('III', 'IV')): return 1.0\n    if s.startswith(('I', 'II')): return 0.0\n    return np.nan"),
    ("cptac_hr = cox_vs_ref(cptac)\n", "cptac_hr = cox_vs_ref(cptac)\n" + STRAT),
    ("    p_str = (f'Log-rank p = {overall_p:.2e}'\n             if overall_p < 0.001 else f'Log-rank p = {overall_p:.4f}')",
     "    p_str = (f'Log-rank p = {overall_p:.2e}'\n             if overall_p < 0.001 else f'Log-rank p = {overall_p:.4f}')\n"
     "    sp = STRAT_P[panel_label]; p_str += (f'\\nStratified log-rank p = {sp:.2e}' if sp < 0.001 else f'\\nStratified log-rank p = {sp:.4f}')"),
    ("ax_c.set_title('Adjusted HR Summary\\n(age, sex, stage | strat: cancer type)',", "ax_c.set_title('Adjusted HR Summary\\n(age, sex, stage I/II vs III/IV | strat: cancer type)',"),
]
for old, new in P:
    assert src.count(old) == 1, f"anchor count {src.count(old)}: {old[:70]!r}"
    src = src.replace(old, new)
src, n = re.subn(r"^(OUT_TIFF\s*=).*$", lambda m: m.group(1) + " r'" + os.path.join(HERE, "Figure2_unified.tiff").replace("\\", "/") + "'", src, flags=re.M); assert n == 1
src += '''
# ---- numbers out (pairwise models as in the figure + single 4-level model as sensitivity) ----
rows = []
for coh, df, hrd, pp, ps in [('TCGA', tcga, tcga_hr, tcga_p, tcga_p_strat), ('CPTAC-3', cptac, cptac_hr, cptac_p, cptac_p_strat)]:
    for g, r in hrd.items(): rows.append(dict(cohort=coh, model='pairwise vs IH_BG (figure)', group=g, hr=r['hr'], lo=r['lo'], hi=r['hi'], p=r['p']))
    d2 = df[['OS_time','OS','group','age','sex_num','stage_num','cancer_type']].dropna(); d2 = d2[d2.OS_time > 0].copy()
    for g in ['IL_BG','IH_BP','IL_BP']: d2[g] = (d2.group == g).astype(float)
    covs = ['IL_BG','IH_BP','IL_BP','age','stage_num'] + (['sex_num'] if d2.sex_num.std() > 0.05 else [])
    m = CoxPHFitter().fit(d2[['OS_time','OS','cancer_type'] + covs], 'OS_time', 'OS', strata='cancer_type')
    for g in ['IL_BG','IH_BP','IL_BP']:
        s = m.summary.loc[g]; rows.append(dict(cohort=coh, model='single 4-level model (sensitivity)', group=g, hr=s['exp(coef)'], lo=s['exp(coef) lower 95%'], hi=s['exp(coef) upper 95%'], p=s['p']))
    rows.append(dict(cohort=coh, model='log-rank', group='4 groups pooled', hr=np.nan, lo=np.nan, hi=np.nan, p=pp)); rows.append(dict(cohort=coh, model='log-rank', group='4 groups stratified', hr=np.nan, lo=np.nan, hi=np.nan, p=ps))
    rows.append(dict(cohort=coh, model='n', group='analytic n / events', hr=len(d2), lo=int(d2.OS.sum()), hi=np.nan, p=np.nan))
    print(coh, 'group sizes:', df.group.value_counts().to_dict())
R = pd.DataFrame(rows); R.to_csv(r\'''' + os.path.join(HERE, "supp_fig2_numbers.csv").replace("\\", "/") + '''\', index=False); print(R.round(4).to_string())
'''
script = os.path.join(HERE, "run_Untitled1_cell1_unified.py"); open(script, "w", encoding="utf-8").write("import matplotlib\nmatplotlib.use('Agg')\n" + src)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=os.path.join(S, "pylib_cox"), MPLBACKEND="Agg"); t0 = time.time()
with open(os.path.join(HERE, "run.log"), "w", encoding="utf-8") as lf:
    rc = subprocess.run([COX, "-s", script], cwd=SB, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
print(f"rc={rc} {time.time()-t0:.0f}s")
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
im = Image.open(os.path.join(HERE, "Figure2_unified.tiff")).convert("RGB"); im.thumbnail((2400, 2400)); im.save(os.path.join(HERE, "Figure2_unified_preview.png"))
