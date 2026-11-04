"""Unified Cox specification (no penalty, Efron ties, strata = cancer type, age + sex + binary stage + SMA z + PC):
PC1-PC10 screening in both cohorts and the joint SMA + PC6 model (manuscript Figure 2 / Supplementary Figure 5A numbers).
Usage:  python code/04_survival/run_unified_cox.py   (lifelines >= 0.30)"""
import os, sys, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE)); DATA = os.path.join(REPO, "data")
os.environ["BODYCOMP_BASE"] = DATA; os.environ["BODYCOMP_AE_DIR"] = DATA
sys.path.insert(0, HERE); import pc6_cox_spec
out = os.path.join(REPO, "results", "reproduced", "unified_cox"); os.makedirs(out, exist_ok=True)
pc6_cox_spec.main(outdir=out); print("results in", out)
