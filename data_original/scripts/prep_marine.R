
library(phyloseq)
library(tidyverse)

BAC_pruned <- readRDS("original/BAC_pruned.rds")
# See https://github.com/edfadeev/Bact-comm-PS85/blob/master/dataset_preprocess.R
# for code to generate

y <- sample_data(BAC_pruned)$leucine
imiss <- which(is.na(y))

tax <- BAC_pruned@tax_table@.Data
colnames(tax)[1] <- "Kingdom"

# replace "unclassified" with the appropriate blank tag
blank <- paste0(c("k", "p", "c", "o", "f", "g"), "__")
tax[str_detect(tax, "_unclassified")] <- "Unclassified"
for (i in 1:6) tax[tax[, i] == "Unclassified", i] <- blank[i]

# make it so labels are unique
for (i in seq(2, 6)) {
  # add a number when the type is unknown... e.g. "g__"
  ii <- nchar(tax[, i]) == 3
  if (sum(ii) > 0)
    tax[ii, i] <- paste0(tax[ii, i], 1:sum(ii))
}
# cumulative labels are harder to read but easier to work with:
for (i in 2:6) {
  tax[, i] <- paste(tax[, i-1], tax[, i], sep = ";")
}
tax <- as.data.frame(tax, stringsAsFactors = TRUE)

# Shimeng: create output files
data_X <- as.data.frame(BAC_pruned@otu_table@.Data[, -imiss])
data_X <- rowsum(data_X, tax$Genus, reorder = TRUE) # p x n matrix
rownames(data_X) <- sort(unique(tax$Rank7))
data_y <- y[-imiss]
(dim(data_X)) # 25909 882 p x n
(length(data_y)) # 882

# write to the subfolder under data_original
write.csv(data_X, "marine_X.csv", row.names = TRUE)
write.csv(data_y, "marine_y.csv", row.names = TRUE)
