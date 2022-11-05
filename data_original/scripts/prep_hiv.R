# Note: assume working in the data_processed/scripts directory
setwd("../../data_original/sCD14_HIV")

library(tidyverse)
library(phyloseq)

# Data provided as phyloseq object by Rivera-Pinto (personal communication)

load("original/Data_HIV.RData")
# Less extreme filtering compared to J. Rivera-Pinto (see procSCD14.R)

# Build a filter (at OTU level)
filter.OTU <- genefilter_sample(x, filterfun_sample(function(x) x >= 1),
                                A = 0.1 * nsamples(x))
# Apply filter
x.filter <- prune_taxa(filter.OTU, x)
tax <- x.filter@tax_table@.Data

# replace "unclassified" with the appropriate blank tag
blank <- paste0(c("k", "p", "c", "o", "f", "g"), "__")
for (i in 1:6) {
  tax[tax[, i] != "unclassified", i] <- paste0(blank[i], tax[tax[, i] != "unclassified", i])
  tax[tax[, i] == "unclassified", i] <- blank[i]
}

# add an OTU column
tax <- cbind(tax, rownames(tax))
colnames(tax)[7] <- "OTU"

# make it so labels are unique
for (i in seq(2, 6)) {
  # add a number when the type is unknown... e.g. "g__"
  ii <- nchar(tax[, i]) == 3
  if (sum(ii) > 0)
    tax[ii, i] <- paste0(tax[ii, i], 1:sum(ii))
}

tax <- as.data.frame(tax)

# # add a root node that combines the three kingdoms into a tree:
# tax <- as.data.frame(tax)
# tax$Rank0 <- rep("Life", nrow(tax))
# tax <- tax[, c(8, 1:7)]

# cumulative labels are harder to read but easier to work with:
for (i in 2:ncol(tax)) {
  tax[, i] <- paste(tax[, i-1], tax[, i], sep = ";")
}
# convert all columns from character to factors for tax_table_to_phylo
for (i in seq_along(tax)) tax[, i] <- factor(tax[, i])

# Shimeng: create output files
y <- sample_data(x.filter)$sCD14
yy <- as.numeric(levels(y))[y]
imiss <- which(is.na(yy)) # four samples have missing sCD14
data_X <- as.data.frame(x.filter@otu_table@.Data[, -imiss])
# Note: see `rowsum`
# reorder if TRUE, then the result will be in order of sort(unique(group)),
data_X <- rowsum(data_X, tax$Genus, reorder=TRUE) # p x n matrix
rownames(data_X) <- sort(unique(tax$Genus))
data_y <- yy[-imiss]
(dim(data_X)) # 282 152 p x n
(length(data_y)) # 152

write.csv(data_X, "hiv_X.csv", row.names = TRUE)
write.csv(data_y, "hiv_y.csv", row.names = TRUE)
