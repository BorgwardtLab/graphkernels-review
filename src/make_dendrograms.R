fix_row_names <- function(data) {

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

  return(data)
}

data <- read.table(
  '../results/Accuracies.csv',
  sep=',',
  header=TRUE,
  row.names=1
)

data <- fix_row_names(data)

distances.accuracy <- dist(data, method='euclidean')
hc <- hclust(distances.accuracy)
hcd <- as.dendrogram(hc)

pdf('Dendrogram_accuracies.pdf')

plot(hcd)

dev.off()

data <- read.table(
  '../results/Distances_Hamming.csv',
  sep=',',
  header=TRUE,
  row.names=1
)

data <- data[, !(colnames(data) %in% c("XX"))]
data <- data[!(rownames(data) %in% c("XX")), ]
#data <- data[!(names(data) %in% c("XX")), ]

data <- fix_row_names(data)
data <- as.dist(data)

distances.hamming <- data
hc <- hclust(distances.hamming)
hcd <- as.dendrogram(hc)

pdf('Dendrogram_Hamming.pdf')

plot(hcd)

dev.off()

