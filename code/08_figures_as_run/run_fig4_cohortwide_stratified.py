"""Fig 4, option (a) decided 2026-11-04: submitted layout and group definition (TCGA cohort-wide medians, TCGA only) kept;
panel A shows the pooled log-rank p (as submitted) AND the cancer-type-stratified log-rank p (Reviewer 2-8). Panels B-D are
unchanged (TCGA data, PC6 and SMA are identical to the submission). The submitted cell (Untitled3 cell 4) is copied into the
FINAL v2 sandbox with only the p-string patched; the notebook is untouched. A stratified, adjusted Cox table for the same four
groups (reference PC6-Low/SMA-High) is computed separately for the Results text."""
import os, re, subprocess, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R1 = os.path.dirname(HERE)
S = r"C:\Users\nsw\AppData\Local\Temp\claude\c--Users-nsw-Desktop-body-composition\61cfca76-a232-4ce7-a356-6d92f663b762\scratchpad"
SB = os.path.join(S, "repro_final_v2").replace("\\", "/"); COX = r"C:\Users\nsw\anaconda3\envs\cox_env\python.exe"
src = open(os.path.join(S, "nb", "Untitled3_cell4.py"), encoding="utf-8").read()
for old in [r"C:\Users\nsw\Desktop\body_composition", r"C:\Users\nsw\body_com_paper_code"]: src = src.replace(old, SB)
OLD = ("lrp = multivariate_logrank_test(merged['OS_months'], merged['group4'], merged['OS']).p_value\n"
       "p_str = f'Log-rank  p = {lrp:.2e}' if lrp < 1e-3 else f'Log-rank  p = {lrp:.4f}'")
NEW = ("def _stratified_logrank(df, T='OS_months', E='OS', G='group4', S_='cancer_type'):  # R2-8\n"
       "    from scipy.stats import chi2 as _chi2\n"
       "    groups = sorted(df[G].unique()); k = len(groups); OE = np.zeros(k); V = np.zeros((k, k))\n"
       "    for _, d in df.groupby(S_):\n"
       "        for t in np.sort(d.loc[d[E] == 1, T].unique()):\n"
       "            at = d[d[T] >= t]; n = len(at); dead = d[(d[T] == t) & (d[E] == 1)]; dt = len(dead)\n"
       "            if n < 2 or dt == 0: continue\n"
       "            nj = np.array([(at[G] == g).sum() for g in groups], float); dj = np.array([(dead[G] == g).sum() for g in groups], float)\n"
       "            p = nj / n; OE += dj - dt * p; V += dt * (n - dt) / (n - 1) * (np.diag(p) - np.outer(p, p))\n"
       "    oe, v = OE[:-1], V[:-1, :-1]; stat = float(oe @ np.linalg.pinv(v) @ oe); return float(_chi2.sf(stat, k - 1))\n"
       "lrp = multivariate_logrank_test(merged['OS_months'], merged['group4'], merged['OS']).p_value\n"
       "lrs = _stratified_logrank(merged)\n"
       "_f = lambda p: f'{p:.2e}' if p < 1e-3 else f'{p:.4f}'\n"
       "p_str = f'Log-rank  p = {_f(lrp)}\\nStratified by cancer type  p = {_f(lrs)}'\n"
       "print(f'FIG4A pooled log-rank p={lrp:.3e}; stratified p={lrs:.3e}; groups={merged.group4.value_counts().to_dict()}')")
assert src.count(OLD) == 1, "p_str anchor"; src = src.replace(OLD, NEW)
OUT_TIFF = os.path.join(HERE, "Figure4_submitted_layout_stratified_p.tiff").replace("\\", "/")
src, n = re.subn(r"^(OUT_TIFF\s*=).*$", lambda m: m.group(1) + " r'" + OUT_TIFF + "'", src, flags=re.M); assert n == 1
script = os.path.join(HERE, "run_Untitled3_cell4_cohortwide_stratified.py"); open(script, "w", encoding="utf-8").write("import matplotlib\nmatplotlib.use('Agg')\n" + src)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=os.path.join(S, "pylib_cox"), MPLBACKEND="Agg"); t0 = time.time()
with open(os.path.join(HERE, "run.log"), "w", encoding="utf-8") as lf:
    rc = subprocess.run([COX, "-s", script], cwd=SB, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
print(f"cell rc={rc} {time.time()-t0:.0f}s"); print([l for l in open(os.path.join(HERE, "run.log"), encoding="utf-8").read().splitlines() if l.startswith("FIG4A") or "DEG:" in l])

# ---- stratified adjusted Cox for the same four groups (cohort-wide medians), FINAL v2 inputs, unified spec
from lifelines import CoxPHFitter
FIN = os.path.join(R1, "final_v2", "inputs")
def stage_bin(x):
    s = str(x).upper()
    if "[" in s or "UNKNOWN" in s or s.strip() in ("", "NAN", "NONE"): return np.nan
    if "III" in s or "IV" in s: return 1.0
    if "II" in s or "I" in s: return 0.0
    return np.nan
rows = []
for coh, clinf, pcf, cut in [("TCGA", "TCGA_with_genes.csv", "ae_latent_pca10_all.csv", True), ("CPTAC-3", "External_with_genes.csv", "ae_latent_pca10_external.csv", False)]:
    cl = pd.read_csv(os.path.join(FIN, clinf)); pc = pd.read_csv(os.path.join(FIN, pcf))
    cl["pid"] = cl.patient_id.astype(str).str[:12] if cut else cl.patient_id.astype(str); pc["pid"] = pc.patient_id.astype(str).str[:12] if cut else pc.patient_id.astype(str)
    d = cl.merge(pc[["pid", "AE_PC6"]], on="pid"); d["PC6_z"] = (d.AE_PC6 - d.AE_PC6.mean()) / d.AE_PC6.std()
    d["OS_months"] = pd.to_numeric(d.OS_time, errors="coerce") / 30.44; d["OS"] = pd.to_numeric(d.OS, errors="coerce"); d["age"] = pd.to_numeric(d.age, errors="coerce")
    d["SMA_zscore"] = pd.to_numeric(d.SMA_zscore, errors="coerce"); d = d.dropna(subset=["PC6_z", "SMA_zscore", "OS_months", "OS"])
    pm, sm = d.PC6_z.median(), d.SMA_zscore.median()   # cohort-wide medians, as submitted
    d["g"] = np.where(d.PC6_z >= pm, "PC6-High", "PC6-Low") + " / " + np.where(d.SMA_zscore >= sm, "SMA-High", "SMA-Low")
    d["sex_bin"] = (d.sex.astype(str).str.upper() == "MALE").astype(float); d["stage_bin"] = d.stage.apply(stage_bin)
    u = d.dropna(subset=["age", "stage_bin"]); u = u[u.OS_months > 0]
    X = pd.get_dummies(u.g, drop_first=False).astype(float).drop(columns=["PC6-Low / SMA-High"]); X.columns = [c.replace(" / ", "__").replace("-", "_") for c in X.columns]
    df = pd.concat([u[["OS_months", "OS", "cancer_type", "age", "sex_bin", "stage_bin"]], X], axis=1)
    cph = CoxPHFitter(); cph.fit(df, "OS_months", "OS", strata="cancer_type"); s = cph.summary
    for g in ["PC6-Low / SMA-Low", "PC6-High / SMA-High", "PC6-High / SMA-Low"]:
        k = g.replace(" / ", "__").replace("-", "_"); r = s.loc[k]
        rows.append(dict(cohort=coh, group=g, n=int((u.g == g).sum()), n_total=len(u), events=int(u.OS.sum()), HR=r["exp(coef)"], lo=r["exp(coef) lower 95%"], hi=r["exp(coef) upper 95%"], p=r["p"]))
        print(f"  [{coh}] {g:22s} n={int((u.g == g).sum()):3d}  HR {r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}-{r['exp(coef) upper 95%']:.2f}) p={r['p']:.3g}   (ref PC6-Low / SMA-High n={int((u.g=='PC6-Low / SMA-High').sum())}; model n={len(u)}/{int(u.OS.sum())})")
pd.DataFrame(rows).to_csv(os.path.join(HERE, "fig4a_stratified_adjusted_cox_cohortwide_medians.csv"), index=False)
# png preview
try:
    from PIL import Image; im = Image.open(OUT_TIFF); im.thumbnail((2400, 2400)); im.convert("RGB").save(OUT_TIFF.replace(".tiff", "__preview.png")); print("preview saved")
except Exception as e: print("preview failed:", e)
