# Note: assume working in the data_processed/scripts directory
setwd("../../data_original/AmericanGut")

# S.H.: this script is adapted from the first two scripts in 
# https://github.com/jacobbien/trac-reproducible/tree/main/AmericanGut
# i.e., 0create_phyloseq_object.R and 1prep_data_all_levels.R
# Currently it is run first to create the csv files in the AmericanGut folder

# BiocManager::install('phyloseq')

# Import and prune American Gut Project (AGP) data    
# McDonald et al. (2018)

library(phyloseq)

# import biom file (can take a few minutes due to large file size):
ag <- import_biom("original/8237_analysis.biom") 
# import metadata from mapping file:
map <- read.delim("original/8237_analysis_mapping.txt", 
                  sep = "\t",
                  header = TRUE, 
                  row.names = 1)
# assign metadata to phyloseq object:
sample_data(ag) <- map

# All fecal data
ag.fe <- subset_samples(ag, body_habitat == "UBERON:feces") ## only fecal samples
# Shimeng: subset samples by race
# ag.fe <- subset_samples(ag.fe, race == "Asian or Pacific Islander")
# Shimeng: subset samples by country
schengen <- c("Spain", "Germany", "Netherlands", "Switzerland",
              "Sweden", "France", "Ireland", "Norway", "Belgium",
              "Italy", "Czech Republic", "Denmark", "Slovakia", "Serbia",
              "Portugal", "Austria", "Greece", "Finland", "Latvia", "Poland")
rest_of_world <- setdiff(map$country, 
                         c(schengen, "USA", "Canada", "Australia", "United Kingdom"))
# setdiff(map$country, schengen)
ag.fe <- subset_samples(ag.fe, country %in% rest_of_world)

ag.fe
## Full fecal phyloseq object
# phyloseq-class experiment-level object
# otu_table()   OTU Table:         [ 27116 taxa and 8440 samples ]
# sample_data() Sample Data:       [ 8440 samples by 517 sample variables ]
# tax_table()   Taxonomy Table:    [ 27116 taxa by 7 taxonomic ranks ]

## Prune samples
depths <- colSums(ag.fe@otu_table@.Data) ## calculate sequencing depths

## Pruning (Minimum sequencing depth: at least 10000 reads per sample)
ag.filt1 <- prune_samples(depths > 10000, ag.fe) 
ag.filt1
# phyloseq-class experiment-level object
# otu_table()   OTU Table:         [ 27116 taxa and 7203 samples ]
# sample_data() Sample Data:       [ 7203 samples by 517 sample variables ]
# tax_table()   Taxonomy Table:    [ 27116 taxa by 7 taxonomic ranks ]

## Pruning (taxa present in at least 10% of samples)
# freq <- rowSums(sign(ag.filt1@otu_table@.Data))
# ag.filt2 <- prune_taxa(freq > 0.1 * nsamples(ag.filt1), ag.filt1) 
# ag.filt2
# phyloseq-class experiment-level object
# otu_table()   OTU Table:         [ 1387 taxa and 7203 samples ]
# sample_data() Sample Data:       [ 7203 samples by 517 sample variables ]
# tax_table()   Taxonomy Table:    [ 1387 taxa by 7 taxonomic ranks ]

# below based on `1prep_data_all_levels.R`
# agp <- ag.filt2
agp <- ag.filt1
# Shimeng: slight modified due to file reading
y_factor <- factor(sample_data(agp)$bmi)
y <- as.numeric(levels(y_factor)[y_factor])
summary(y)
# filter 16 - 40 BMI (which excludes the most extreme categories
# of thinness and obesity. Note that those categories do not have
# lower/upper bounds)
# https://web.archive.org/web/20090418181049/http://www.who.int/bmi/index.jsp?introPage=intro_3.html
keep <- which(!is.na(y) & y <= 40 & y >= 16)
tax <- agp@tax_table@.Data
# tax <- cbind("Life", tax); colnames(tax)[1] <- "Rank0"
# add an OTU column
tax <- cbind(tax, rownames(tax))
colnames(tax)[ncol(tax)] <- "OTU"

# make it so labels are unique
for (i in seq(2, 8)) {
  # add a number when the type is unknown... e.g. "g__"
  ii <- nchar(tax[, i]) == 3
  if (sum(ii) > 0)
    tax[ii, i] <- paste0(tax[ii, i], 1:sum(ii))
}
# cumulative labels are harder to read but easier to work with:
for (i in 2:8) {
  tax[, i] <- paste(tax[, i-1], tax[, i], sep = ";")
}
tax <- as.data.frame(tax, stringsAsFactors = TRUE)

# Shimeng: create output files
data_X <- as.data.frame(agp@otu_table@.Data[, keep])
data_X <- rowsum(data_X, tax$Rank7, reorder = TRUE) # p x n matrix
rownames(data_X) <- sort(unique(tax$Rank7))
data_y <- y[keep]
(dim(data_X)) # 25909 882 p x n
(length(data_y)) # 882

# write to the subfolder under data_original
write.csv(data_X, "ag_X_rest.csv", row.names = TRUE)
write.csv(data_y, "ag_y_rest.csv", row.names = TRUE)

