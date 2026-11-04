"""FINAL re-run: corrected clinical inputs (TCGA stage, CPTAC SMA) + quantile-mapped CPTAC PCs (route B)
+ unified Cox specification (no penalizer; missing stage -> NaN -> dropped). Notebook cells are copied into
repro_final/ with path redirection and the minimal patches listed in PATCHES; the original notebooks are untouched."""
import os, subprocess, sys, time

S = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(S, "nb")
SB = os.path.join(S, "repro_final_v2").replace("\\", "/")
LOG = os.path.join(S, "repro_final_v2_logs"); os.makedirs(LOG, exist_ok=True)
COX = r"C:\Users\nsw\anaconda3\envs\cox_env\python.exe"
MULO = r"C:\Users\nsw\anaconda3\envs\mul-o\python.exe"

CLEAN_STAGE_OLD = ("def clean_stage(s):\n    if pd.isna(s): return np.nan\n    s = str(s).upper()\n    if 'IV'  in s: return 4")
CLEAN_STAGE_NEW = ("def clean_stage(s):\n    if pd.isna(s): return np.nan\n    s = str(s).upper()\n"
                   "    if '[' in s or 'UNKNOWN' in s or s.strip() in ('', 'NAN', 'NONE'): return np.nan  # unified spec: missing stage\n"
                   "    if 'IV'  in s: return 4")
FSB_OLD = 'def format_stage_binary(x):\n    s = str(x).upper()\n    if ("III" in s) or ("IV" in s): return 1.0'
FSB_NEW = ('def format_stage_binary(x):\n    s = str(x).upper()\n'
           "    if '[' in s or 'UNKNOWN' in s or s.strip() in ('', 'NAN', 'NONE'): return np.nan  # unified spec: missing stage\n"
           '    if ("III" in s) or ("IV" in s): return 1.0')
SUPP3_STAGE_OLD = ("df['stage_bin'] = df['stage'].apply(\n    lambda s: 1 if str(s).strip().upper() in ['III','IV'] else 0)")
SUPP3_STAGE_NEW = ("def format_stage_binary(x):  # unified spec\n"
                   "    s = str(x).upper().replace('STAGE', '').strip()\n"
                   "    if s in ('', 'NAN', 'NONE', 'UNKNOWN', '[DISCREPANCY]', '[NOT AVAILABLE]', '[NOT APPLICABLE]'): return np.nan\n"
                   "    if s.startswith(('III', 'IV')): return 1.0\n"
                   "    if s.startswith(('I', 'II')): return 0.0\n"
                   "    return np.nan\n"
                   "df['stage_bin'] = df['stage'].apply(format_stage_binary)")
PEN_OLD, PEN_NEW = "CoxPHFitter(penalizer=0.1)", "CoxPHFitter()  # unified spec: no penalizer"

PATCHES = {
    "Untitled1_cell0": [(CLEAN_STAGE_OLD, CLEAN_STAGE_NEW)],
    "Untitled1_cell1": [(CLEAN_STAGE_OLD, CLEAN_STAGE_NEW)],
    "Untitled1_cell2": [(PEN_OLD, PEN_NEW)],
    "Untitled3_cell1": [(SUPP3_STAGE_OLD, SUPP3_STAGE_NEW), (PEN_OLD, PEN_NEW)],
    "Untitled3_cell3": [(FSB_OLD, FSB_NEW), (PEN_OLD, PEN_NEW)],
}
CELLS = [
    ("Untitled1_cell0", COX, "pylib_cox", "Supp Fig 1 independence -> Figure1.tiff"),
    ("Untitled1_cell1", COX, "pylib_cox", "Supp Fig 2 4-group survival -> Figure2.tiff"),
    ("Untitled1_cell2", COX, "pylib_cox", "Supp Fig 5A PC selection -> Figure3.tiff"),
    ("Untitled1_cell3", COX, "pylib_cox", "clinical_impact_pc6 CSVs (Breslow, not in paper)"),
    ("Untitled3_cell1", COX, "pylib_cox", "Supp Fig 3 SMA cutoff sensitivity"),
    ("Untitled3_cell2", COX, "pylib_cox", "Supp Fig 8 PC6 distribution"),
    ("Untitled3_cell3", COX, "pylib_cox", "Fig 2 clinical impact -> Figure4_fixed.tiff"),
    ("Untitled3_cell4", COX, "pylib_cox", "Fig 4 joint effects -> Figure6.tiff"),
    ("Untitled3_cell5", COX, "pylib_cox", "Supp Fig 6 gene-level corr -> FigureS_GeneLevelCorr.tiff"),
    ("Untitled3_cell6", COX, "pylib_cox", "Fig 3 Warburg -> Figure5.tiff + results_ssgsea_pc6.csv"),
    ("Untitled1_cell9", MULO, "pylib_mulo", "Supp Fig 5B weight-based enrichment of 10 PCs"),
]
only = set(sys.argv[1:])
for cell, py, lib, what in CELLS:
    if only and cell not in only:
        continue
    src = open(os.path.join(NB, cell + ".py"), encoding="utf-8").read()
    for old in [r"C:\Users\nsw\Desktop\body_composition", r"C:\Users\nsw\body_com_paper_code"]:
        src = src.replace(old, SB)
    src = src.replace("np.trapz(", "np.trapezoid(")
    for old, new in PATCHES.get(cell, []):
        assert src.count(old) == 1, f"{cell}: patch anchor count={src.count(old)}: {old[:50]!r}"
        src = src.replace(old, new)
    script = os.path.join(SB, f"run_{cell}.py")
    open(script, "w", encoding="utf-8").write("import matplotlib\nmatplotlib.use('Agg')\n" + src)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=os.path.join(S, lib), MPLBACKEND="Agg")
    t0 = time.time()
    with open(os.path.join(LOG, cell + ".log"), "w", encoding="utf-8") as lf:
        rc = subprocess.run([py, "-s", script], cwd=SB, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    print(f"{cell:16s} rc={rc}  {time.time()-t0:6.0f}s  patches={len(PATCHES.get(cell, []))}  {what}", flush=True)
print("ALL DONE", flush=True)
