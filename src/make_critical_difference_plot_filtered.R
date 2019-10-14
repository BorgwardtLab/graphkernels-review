data <- read.table(
  '../results/Accuracies_filtered.csv',
  sep=',',
  header=TRUE,
  row.names=1
)

# Graphlet kernel
rownames(data) <- gsub("GL", "Graphlet", rownames(data))

# Hash Graph Kernels
rownames(data) <- gsub("HGKSP_seed0", "HGK-SP", rownames(data))
rownames(data) <- gsub("HGKWL_seed0", "HGK-WL", rownames(data))

# Histogram kernels
rownames(data) <- gsub("EH", "Histogram (E)", rownames(data))
rownames(data) <- gsub("VH", "Histogram (V)", rownames(data))

# Weisfeiler--Lehman Optimal Assignment kernel
rownames(data) <- gsub("WLOA", "WL-OA", rownames(data))

library('scmamp')

pdf('Critical_difference_plot.pdf')
plotCD(t(data), alpha=0.05)
