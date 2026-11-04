# Code

Two kinds of scripts are provided.

**Reproducible from this repository** (paths are relative to the repository root; tested on the data shipped here):

| script | what it reproduces | environment |
|---|---|---|
| `03_projection/project_cptac.py` | CPTAC-3 PC1–PC10 scores from the frozen autoencoder (quantile-mapped projection, `data/ae_latent_pca10_external.csv`) | `environment/requirements-torch.txt` |
| `04_survival/run_unified_cox.py` (+ `pc6_cox_spec.py`) | PC1–PC10 screening and the joint SMA + PC6 Cox models (Figure 2, Supplementary Figure 5A) | `environment/requirements-analysis.txt` |
| `05_figure4/make_fig4_final_layout.py` | Figure 4 (A–D) and Supplementary Figure 9, with all numbers | `environment/requirements-analysis.txt` |

**As run for the paper** (copied verbatim; they contain the absolute paths of the analysis computer and the names of the
original input files, which are documented in the top-level README):

- `01_gep_score/gep.R` — T-cell-inflamed GEP score (per-gene z-score of linear expression, mean over genes).
- `02_autoencoder/autoe.py` — gene selection (ANOVA on the training partition), scaler, autoencoder training (64-d bottleneck), PCA; produces `model/ae_v3_outputs/`.
- `03_projection/09b_project_final_v2_as_run.py` — original projection script (same computation as `project_cptac.py`).
- `06_deg_limma/` — `prep_deg.py` (expression matrix + group labels), `limma_deg.R` (limma, with and without cancer type as covariate), `post_deg.py` (adjusted signature/marker summaries).
- `07_hallmark_weight_enrichment/` — weight-based Hallmark enrichment of PC1–PC10 (Supplementary Figure 5B) and the ssGSEA / gene-level analyses.
- `08_figures_as_run/` — the figure notebooks (`notebooks/`), the extracted notebook cells (`notebook_cells/`), the driver `run_final_v2.py` (which lists the exact patches applied to the submitted cells: no Cox penalty, explicit missing-stage coding) and the supplementary-figure scripts.
