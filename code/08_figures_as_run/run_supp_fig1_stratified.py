"""Supp Fig 1 (Untitled1_cell0) re-run on FINAL v2 with panel D moved to the UNIFIED Cox spec:
stratified by cancer type, binary stage (I/II vs III/IV; missing -> excluded), no penalizer, no robust SE — the spec the Methods text
already claims ("stratified by cancer type") but the submitted cell did not implement (unstratified, stage 1-4 continuous, robust SE).
Panels A-C unchanged. Sandbox = scratchpad/repro_final_v2; notebook untouched. Outputs: Figure1_stratified.tiff/png, supp_fig1D_cox_stratified.csv"""
import os, re, subprocess, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S = r"C:\Users\nsw\AppData\Local\Temp\claude\c--Users-nsw-Desktop-body-composition\61cfca76-a232-4ce7-a356-6d92f663b762\scratchpad"
SB = os.path.join(S, "repro_final_v2").replace("\\", "/"); COX = r"C:\Users\nsw\anaconda3\envs\cox_env\python.exe"
src = open(os.path.join(S, "nb", "Untitled1_cell0.py"), encoding="utf-8").read()
for old in [r"C:\Users\nsw\Desktop\body_composition", r"C:\Users\nsw\body_com_paper_code"]: src = src.replace(old, SB)
P = [
    ("def clean_stage(s):\n    if pd.isna(s): return np.nan\n    s = str(s).upper()\n    if 'IV'  in s: return 4\n    if 'III' in s: return 3\n    if 'II'  in s: return 2\n    if 'I'   in s: return 1\n    return np.nan",
     "def clean_stage(s):  # unified spec: binary stage, missing/unknown -> NaN (excluded)\n    s = str(s).upper().replace('STAGE', '').strip()\n"
     "    if s in ('', 'NAN', 'NONE', 'UNKNOWN', '[DISCREPANCY]', '[NOT AVAILABLE]', '[NOT APPLICABLE]'): return np.nan\n"
     "    if s.startswith(('III', 'IV')): return 1.0\n    if s.startswith(('I', 'II')): return 0.0\n    return np.nan"),
    ("    df2 = df[['OS_time','OS','GEP_score','SMA_zscore',\n               'age','sex_num','stage_num']].dropna()",
     "    df2 = df[['OS_time','OS','GEP_score','SMA_zscore',\n               'age','sex_num','stage_num','cancer_type']].dropna()"),
    ("    fit_df = df2[['OS_time','OS'] + covs].copy()\n    cph = CoxPHFitter()\n    cph.fit(fit_df, duration_col='OS_time', event_col='OS', robust=True)",
     "    fit_df = df2[['OS_time','OS','cancer_type'] + covs].copy()\n    cph = CoxPHFitter()  # unified spec: no penalizer, stratified by cancer type\n    cph.fit(fit_df, duration_col='OS_time', event_col='OS', strata='cancer_type')\n"
     "    print(f'[{name}] stratified Cox n={len(fit_df)} events={int(fit_df.OS.sum())}')"),
    ("ax_d.set_title('Adjusted Cox Model\\n(age, sex, stage)', fontsize=TIS, fontweight='bold')",
     "ax_d.set_title('Adjusted Cox Model\\n(age, sex, stage; stratified by cancer type)', fontsize=TIS, fontweight='bold')"),
    ("ax_d.set_xlim(0.2, 2.6)", "ax_d.set_xlim(0.3, 3.7)  # widened: CPTAC SMA x GEP upper CI 3.29"),
]
for old, new in P:
    assert src.count(old) == 1, f"anchor count {src.count(old)}: {old[:70]!r}"
    src = src.replace(old, new)
src, n = re.subn(r"^(OUT_TIFF\s*=).*$", lambda m: m.group(1) + " r'" + os.path.join(HERE, "Figure1_stratified.tiff").replace("\\", "/") + "'", src, flags=re.M); assert n == 1
src += "\ndf_cox.to_csv(r'" + os.path.join(HERE, "supp_fig1D_cox_stratified.csv").replace("\\", "/") + "', index=False)\n"
src += "df_cor.to_csv(r'" + os.path.join(HERE, "supp_fig1C_per_type_spearman.csv").replace("\\", "/") + "', index=False)\nprint(df_cox.to_string()); print(df_cor.to_string())\n"
script = os.path.join(HERE, "run_Untitled1_cell0_stratified.py"); open(script, "w", encoding="utf-8").write("import matplotlib\nmatplotlib.use('Agg')\n" + src)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=os.path.join(S, "pylib_cox"), MPLBACKEND="Agg"); t0 = time.time()
with open(os.path.join(HERE, "run.log"), "w", encoding="utf-8") as lf:
    rc = subprocess.run([COX, "-s", script], cwd=SB, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
print(f"rc={rc} {time.time()-t0:.0f}s")
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
im = Image.open(os.path.join(HERE, "Figure1_stratified.tiff")).convert("RGB"); im.thumbnail((2400, 2400)); im.save(os.path.join(HERE, "Figure1_stratified_preview.png"))
