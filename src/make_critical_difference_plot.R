data <- read.table(
  '../results/auroc.csv',
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
rownames(data) <- gsub("EH_gkl", "Histogram (E)", rownames(data))
rownames(data) <- gsub("VH", "Histogram (V)", rownames(data))

# Weisfeiler--Lehman Optimal Assignment kernel
rownames(data) <- gsub("WLOA", "WL-OA", rownames(data))

# Shortest path
rownames(data) <- gsub("SP_gkl", "SP", rownames(data)) 

# RW kernel
rownames(data) <- gsub("RW_gkl", "RW", rownames(data)) 

# Subgraph Matching
rownames(data) <- gsub("CSM_gkl", "CSM", rownames(data)) 

library('scmamp')

pdf('Critical_difference_plot.pdf', width=7.5)
plotCD(t(data), alpha=0.05)
