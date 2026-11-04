# tumor-host-pc6 — Tumor metabolic program (PC6) × host skeletal muscle in TCGA and CPTAC-3

Code, derived data and the trained autoencoder for the manuscript *"[title]"* (Scientific Reports, revision 1; authors listed in the manuscript).
Discovery cohort: TCGA (n = 495, eight cancer types); external cohort: CPTAC-3 (n = 179; kidney, pancreas, uterus).

## What is here

| folder | content |
|---|---|
| `data/` | per-patient tables used by every analysis: `TCGA_with_genes.csv`, `External_with_genes.csv` (clinical variables, L3 body composition from iAID Image — SMA, SMD, NAMA, LAMA, VAT, SAT —, SMA z-scores and tertiles, GEP score, immune–host subgroup), `ae_latent_pca10_all.csv` / `ae_latent_pca10_external.csv` (PC1–PC10 scores), `ex_GEP_score.csv`, `cptac_rna_file_selection_ruleM_181.csv` (GDC file UUIDs selected per CPTAC patient), `ae_gene_weights_all_PCs.csv` (encoder-derived gene weights per PC) |
| `data/expression/` | `tcga_expr_log2_495.csv.gz` (log2(RSEM+1), 495 patients), `cptac_expr_tpm_179.csv.gz` (STAR TPM, primary tumour files), `tcga_quantile_reference_3000genes.npz` (sorted TCGA values of the 3,000 model genes, used for the external projection) |
| `data/deg/` | limma results (with / without cancer type as covariate) and the group labels |
| `data/gene_sets/` | the 3,000 autoencoder input genes, GEP genes, marker sets, Figure 4C signatures, MSigDB Hallmark |
| `model/ae_v3_outputs/` | `ae_model_bundle.pt` (PyTorch state dict + scaler + gene list + PCA mean/components), training summary, bottleneck sensitivity |
| `results/tables/` | the numbers behind Figures 2–4 and Supplementary Figures 1–9 |
| `results/figures_png/` | the final figures (preview resolution) |
| `code/` | analysis code; see `code/README.md` |
| `environment/` | package versions |

## Data sources (not redistributed in full)

- TCGA: RNA-seq (PanCanAtlas, EB++ adjusted RSEM, log2(x+1)) and clinical data (TCGA-CDR) from the GDC, https://portal.gdc.cancer.gov; CT images from TCIA, https://www.cancerimagingarchive.net.
- CPTAC-3: STAR TPM expression and clinical data from the GDC (primary-tumour files listed in `data/cptac_rna_file_selection_ruleM_181.csv`); CT images from TCIA (CPTAC collections).
- Body composition was measured on the L3 slice with iAID Image (commercial software; outputs are in the per-patient tables).

## Reproducing the main results

```bash
pip install -r environment/requirements-analysis.txt
python code/04_survival/run_unified_cox.py          # Figure 2 / Supp. Fig. 5A Cox models  -> results/reproduced/unified_cox/
python code/05_figure4/make_fig4_final_layout.py    # Figure 4 + Supp. Fig. 9               -> results/reproduced/figure4/
pip install -r environment/requirements-torch.txt
python code/03_projection/project_cptac.py          # CPTAC PC scores from the frozen model  -> results/reproduced/
```

Model: 3,000 genes (one-way ANOVA across the four immune–host subgroups on the TCGA training partition, 70/15/15 split,
GEP genes excluded) → StandardScaler → autoencoder (hidden 1024-256, 64-d bottleneck, dropout 0.3, early stopping) →
PCA on the 64-d bottleneck of all 495 patients (PC1–PC10, 88.4 % variance). External projection: per-gene quantile mapping
of CPTAC TPM onto the TCGA reference distribution, then the frozen scaler/encoder/PCA. Survival: Cox models without penalty
(Efron ties), stratified by cancer type, adjusted for age, sex and stage (I/II vs III/IV); complete cases.

## Citation and license

Please cite the manuscript (reference to be added on publication). Code: MIT (`LICENSE`); derived data: CC BY 4.0 (`DATA_LICENSE.md`).
Archived at Zenodo: DOI to be added.
