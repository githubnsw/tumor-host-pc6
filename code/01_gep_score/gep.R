library(MCPcounter)
library(data.table)

# ╔══════════════════════════════════════════════════════════════╗
# ║                    ⚙️ 설정 (파일 경로 수정!)                    ║
# ╚══════════════════════════════════════════════════════════════╝

# 입력 파일 경로 (본인 파일로 수정)
INPUT_FILE <- "merged_expression_181.csv"

# 출력 폴더
OUTPUT_DIR <- "./"

# ╔══════════════════════════════════════════════════════════════╗
# ║                    GEP 유전자 정의                             ║
# ╚══════════════════════════════════════════════════════════════╝

# T-cell Inflamed GEP (18-gene signature) - Ayers et al., JCI 2017
GEP_18_GENES <- c(
" PSMB10", "HLA-DQA1", "HLA-DRB1", "CMKLR1", "HLA-E",
"NKG7", "CD8A", "CCL5", "CXCL9", "CD27", "CXCR6", "IDO1", "STAT1", "CD274", "CD276",
"LAG3", "PDCD1LG2", "TIGIT
")

# 간소화된 GEP (12-gene)
GEP_12_GENES <- c(
  "CXCL9", "CXCL10", "IDO1", "IFNG", "HLA-DRA", "STAT1",
  "GZMB", "PRF1", "LAG3", "PDCD1", "CD274", "CTLA4"
)

# 사용할 GEP 유전자 선택
GEP_GENES <- GEP_18_GENES

# ╔══════════════════════════════════════════════════════════════╗
# ║                    데이터 로드                                 ║
# ╚══════════════════════════════════════════════════════════════╝

cat("=" , rep("=", 59), "\n", sep = "")
cat("GEP Score & MCP-counter 계산\n")
cat("=" , rep("=", 59), "\n", sep = "")

cat("\n[1/4] 데이터 로딩...\n")

# CSV 읽기
expr <- fread(INPUT_FILE)
genes <- expr[[1]]
mat <- as.matrix(expr[, -1, with = FALSE])
rownames(mat) <- genes

# 유전자 이름 대문자로 통일
rownames(mat) <- toupper(rownames(mat))

cat("   Shape:", nrow(mat), "genes x", ncol(mat), "samples\n")

# ╔══════════════════════════════════════════════════════════════╗
# ║                (A) GEP Score 계산                              ║
# ╚══════════════════════════════════════════════════════════════╝

cat("\n[2/4] GEP Score 계산...\n")

GEP_GENES <- toupper(GEP_GENES)

# 매칭되는 유전자 확인
present_genes <- intersect(GEP_GENES, rownames(mat))
missing_genes <- setdiff(GEP_GENES, rownames(mat))

cat("   정의된 GEP 유전자:", length(GEP_GENES), "\n")
cat("   매칭된 유전자:", length(present_genes), "\n")

if (length(missing_genes) > 0) {
  cat("   Missing:", paste(missing_genes, collapse = ", "), "\n")
}

if (length(present_genes) == 0) {
  stop("GEP 유전자가 데이터에 없습니다!")
}

# Z-score 정규화
gep_mat <- mat[present_genes, , drop = FALSE]
z_scores <- t(scale(t(gep_mat)))

# GEP score = 평균 z-score
gep_score <- colMeans(z_scores, na.rm = TRUE)

cat("   GEP Score 범위:", round(min(gep_score), 3), "~", round(max(gep_score), 3), "\n")

# Median으로 High/Low 분류
gep_median <- median(gep_score)
gep_group <- ifelse(gep_score >= gep_median, "Immune_High", "Immune_Low")

cat("   Median cutoff:", round(gep_median, 4), "\n")
cat("   Immune_High:", sum(gep_group == "Immune_High"), "samples\n")
cat("   Immune_Low:", sum(gep_group == "Immune_Low"), "samples\n")

# ╔══════════════════════════════════════════════════════════════╗
# ║                (B) MCP-counter 계산                            ║
# ╚══════════════════════════════════════════════════════════════╝

cat("\n[3/4] MCP-counter 계산...\n")

# MCP-counter 실행
mcp_result <- MCPcounter.estimate(mat, featuresType = "HUGO_symbols")

cat("   MCP-counter 완료\n")
cat("   추정된 세포 타입:", nrow(mcp_result), "\n")

for (cell in rownames(mcp_result)) {
  cat("      -", cell, "\n")
}

# ╔══════════════════════════════════════════════════════════════╗
# ║                    결과 저장                                   ║
# ╚══════════════════════════════════════════════════════════════╝

cat("\n[4/4] 결과 저장...\n")

# 1. GEP Score 저장
gep_df <- data.table(
  sample = names(gep_score),
  GEP_score = as.numeric(gep_score),
  GEP_group = gep_group
)
fwrite(gep_df, paste0(OUTPUT_DIR, "GEP_score.csv"))
cat("   GEP_score.csv 저장 완료\n")

# 2. MCP-counter 저장
mcp_df <- data.table(cell_type = rownames(mcp_result), mcp_result)
fwrite(mcp_df, paste0(OUTPUT_DIR, "MCPcounter_scores.csv"))
cat("   MCPcounter_scores.csv 저장 완료\n")

# 3. 통합 데이터 저장
mcp_t <- as.data.table(t(mcp_result))
mcp_t[, sample := colnames(mcp_result)]
combined <- merge(gep_df, mcp_t, by = "sample")
fwrite(combined, paste0(OUTPUT_DIR, "combined_immune_scores.csv"))
cat("   combined_immune_scores.csv 저장 완료\n")

# ╔══════════════════════════════════════════════════════════════╗
# ║                    시각화                                      ║
# ╚══════════════════════════════════════════════════════════════╝

cat("\n시각화 생성...\n")

pdf(paste0(OUTPUT_DIR, "immune_scores_plot.pdf"), width = 12, height = 8)
par(mfrow = c(2, 2))

# 1. GEP 히스토그램
hist(gep_score, breaks = 30, col = "steelblue", border = "white",
     main = "GEP Score Distribution", xlab = "GEP Score")
abline(v = gep_median, col = "red", lwd = 2, lty = 2)
legend("topright", paste("Median =", round(gep_median, 3)), col = "red", lty = 2, lwd = 2)

# 2. GEP 박스플롯
boxplot(gep_score ~ gep_group, col = c("salmon", "lightblue"),
        main = "GEP Score by Group", ylab = "GEP Score")

# 3. MCP 히트맵
set.seed(42)
sample_idx <- sample(1:ncol(mcp_result), min(50, ncol(mcp_result)))
mcp_subset <- mcp_result[, sample_idx]
image(t(scale(t(mcp_subset))), col = colorRampPalette(c("blue", "white", "red"))(100),
      axes = FALSE, main = "MCP-counter (50 samples)")
axis(2, at = seq(0, 1, length.out = nrow(mcp_result)),
     labels = rownames(mcp_result), las = 2, cex.axis = 0.7)

# 4. GEP vs CD8 T cells
cd8_scores <- mcp_result["T cells", ]
plot(gep_score, cd8_scores, pch = 19, col = adjustcolor("steelblue", 0.5),
     main = "GEP vs T cells (MCP)", xlab = "GEP Score", ylab = "T cells")
abline(lm(cd8_scores ~ gep_score), col = "red", lwd = 2)
cor_val <- cor(gep_score, cd8_scores, use = "complete.obs")
legend("topleft", paste("r =", round(cor_val, 3)), bty = "n")

dev.off()
cat("   immune_scores_plot.pdf 저장 완료\n")

# ╔══════════════════════════════════════════════════════════════╗
# ║                    완료                                        ║
# ╚══════════════════════════════════════════════════════════════╝

cat("\n", rep("=", 60), "\n", sep = "")
cat("분석 완료!\n")
cat(rep("=", 60), "\n", sep = "")

cat("\n생성된 파일:\n")
cat("  1. GEP_score.csv\n")
cat("  2. MCPcounter_scores.csv\n")
cat("  3. combined_immune_scores.csv\n")
cat("  4. immune_scores_plot.pdf\n")
