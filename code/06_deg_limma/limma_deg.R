## A. limma DEG: PC6-High/SMA-Low vs all others, with and without cancer type as covariate, for both group definitions;
## plus per-cancer-type DEG and the direction-concordant intersection. Thresholds as in the submitted analysis: FDR < 0.05 & |logFC| >= 0.5.
suppressMessages({library(limma); library(data.table)})
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
expr <- fread(file.path(here, "expr_log2_tcga.csv")); genes <- expr[[1]]; expr <- as.matrix(expr[, -1]); rownames(expr) <- genes
g <- fread(file.path(here, "groups_tcga.csv")); g <- g[match(colnames(expr), g$pid), ]
stopifnot(all(g$pid == colnames(expr)))
g$cancer_type <- factor(g$cancer_type)
res <- list(); summ <- list()
run_limma <- function(worst, adjust, label) {
  design <- if (adjust) model.matrix(~ worst + cancer_type, data = data.frame(worst = worst, cancer_type = g$cancer_type)) else model.matrix(~ worst)
  fit <- eBayes(lmFit(expr, design), trend = TRUE)
  tt <- topTable(fit, coef = "worst", number = Inf, sort.by = "none")
  tt$gene <- rownames(tt); tt$sig <- tt$adj.P.Val < 0.05 & abs(tt$logFC) >= 0.5
  fwrite(tt[order(tt$adj.P.Val), ], file.path(here, paste0("limma_", label, ".csv")))
  data.frame(analysis = label, n_sig = sum(tt$sig), n_up = sum(tt$sig & tt$logFC > 0), n_down = sum(tt$sig & tt$logFC < 0),
             median_logFC_sig = median(tt$logFC[tt$sig]), n_fdr05_any = sum(tt$adj.P.Val < 0.05))
}
for (def in c("cohortwide", "withintype")) {
  w <- g[[paste0("worst_", def)]]
  summ[[length(summ) + 1]] <- run_limma(w, FALSE, paste0(def, "_unadjusted"))
  summ[[length(summ) + 1]] <- run_limma(w, TRUE,  paste0(def, "_adj_cancer_type"))
}
## per-cancer-type DEG (within-type definition), types with >= 8 worst and >= 8 others
w <- g$worst_withintype; per <- list(); tabs <- list()
for (ct in levels(g$cancer_type)) {
  idx <- g$cancer_type == ct
  if (sum(w[idx] == 1) < 8 || sum(w[idx] == 0) < 8) next
  fit <- eBayes(lmFit(expr[, idx], model.matrix(~ w[idx])), trend = TRUE)
  tt <- topTable(fit, coef = 2, number = Inf, sort.by = "none"); tt$gene <- rownames(tt)
  tt$sig <- tt$adj.P.Val < 0.05 & abs(tt$logFC) >= 0.5
  per[[ct]] <- tt; tabs[[ct]] <- data.frame(cancer_type = ct, n_worst = sum(w[idx] == 1), n_other = sum(w[idx] == 0), n_sig = sum(tt$sig), n_up = sum(tt$sig & tt$logFC > 0), n_down = sum(tt$sig & tt$logFC < 0))
}
pt <- do.call(rbind, tabs); fwrite(pt, file.path(here, "limma_per_type_summary.csv"))
## concordance across types: sign of logFC in every tested type, and number of types where the gene is a DEG
sign_mat <- sapply(per, function(tt) sign(tt$logFC[match(genes, tt$gene)]))
sig_mat  <- sapply(per, function(tt) tt$sig[match(genes, tt$gene)])
conc <- data.frame(gene = genes, n_types_tested = rowSums(!is.na(sign_mat)), n_types_sig = rowSums(sig_mat, na.rm = TRUE),
                   n_up = rowSums(sign_mat > 0, na.rm = TRUE), n_down = rowSums(sign_mat < 0, na.rm = TRUE))
conc$all_same_direction <- conc$n_types_tested >= 2 & (conc$n_up == conc$n_types_tested | conc$n_down == conc$n_types_tested)
adj <- fread(file.path(here, "limma_withintype_adj_cancer_type.csv")); adj_sig <- adj$gene[adj$sig]
conc$adj_model_DEG <- conc$gene %in% adj_sig
fwrite(conc, file.path(here, "limma_per_type_concordance.csv"))
inter <- conc[conc$adj_model_DEG & conc$all_same_direction, ]
inter2 <- conc[conc$n_types_sig >= 2 & conc$all_same_direction, ]
summ[[length(summ) + 1]] <- data.frame(analysis = "adj-model DEGs with the same logFC sign in every tested cancer type", n_sig = nrow(inter), n_up = sum(inter$n_up > 0), n_down = sum(inter$n_down > 0), median_logFC_sig = NA, n_fdr05_any = NA)
summ[[length(summ) + 1]] <- data.frame(analysis = "DEG in >= 2 cancer types separately, concordant direction", n_sig = nrow(inter2), n_up = sum(inter2$n_up > 0), n_down = sum(inter2$n_down > 0), median_logFC_sig = NA, n_fdr05_any = NA)
S <- do.call(rbind, summ); fwrite(S, file.path(here, "limma_summary.csv")); print(S); print(pt)
fwrite(inter[order(-inter$n_types_sig), ], file.path(here, "deg_intersection_adj_and_concordant.csv"))
