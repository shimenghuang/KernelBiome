# Note: assume working in the data_processed/scripts directory
setwd("../../data_original/Tara")

library(phyloseq)

load("original/taraData.rda")

tara  <- physeq

map <- data.frame(metadata, row.names=metadata$SampleName)

sample_data(tara) <- map ## assign metadata to phyloseq object

tara

## Prune samples

badTaxa = c("OTU1") # undefined across all ranks
allTaxa = taxa_names(tara)
allTaxa <- allTaxa[!(allTaxa %in% badTaxa)]
tara = prune_taxa(allTaxa, tara)

depths <- colSums(tara@otu_table@.Data) ## calculate sequencing depths

## Pruning (Minimum sequencing depth: at least 10000 reads per sample)
tara.filt1 <- prune_samples(depths > 10000, tara)
tara.filt1

y <- sample_data(tara.filt1)$Mean_Salinity..PSU.
summary(y)
keep <- which(!is.na(sample_data(tara)$Mean_Salinity..PSU.))

# Longhurst provinces in Longhurst
sample_data(tara)$Marine.pelagic.biomes..Longhurst.2007.

tax <- tara@tax_table@.Data

# replace "unclassified" with the appropriate blank tag
blank <- paste0(c("k", "p", "c", "o", "f", "g", "s"), "__")
for (i in 1:7) tax[tax[, i] == "unclassified", i] <- blank[i]
for (i in 1:7) tax[tax[, i] == "undef", i] <- blank[i]
for (i in 1:7) tax[tax[, i] == "", i] <- blank[i]

# add an OTU column
colnames(tax)[(colnames(tax) == "OTU.rep")] <- "Species"
colnames(tax)[(colnames(tax) == "Domain")] <- "Kingdom"

# make it so labels are unique
for (i in seq(2, 7)) {
  # add a number when the type is unknown... e.g. "g__"
  ii <- nchar(tax[, i]) == 3
  if (sum(ii) > 0)
    tax[ii, i] <- paste0(tax[ii, i], 1:sum(ii))
}

# cumulative labels are harder to read but easier to work with:
for (i in 2:7) {
  tax[, i] <- paste(tax[, i-1], tax[, i], sep = ";")
}
tax <- as.data.frame(tax, stringsAsFactors = TRUE)

# create output files
data_X <- as.data.frame(tara@otu_table@.Data[, keep])
data_X <- rowsum(data_X, tax$Species, reorder = TRUE) # p x n matrix
rownames(data_X) <- sort(unique(tax$Species))
data_y <- y[keep]
(dim(data_X)) # 35650  136
(length(data_y)) # 136

# write to the subfolder under data_original
write.csv(data_X, "tara_X.csv", row.names = TRUE)
write.csv(data_y, "tara_y.csv", row.names = TRUE)
