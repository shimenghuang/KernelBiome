# Note: assume working in this script's directory
setwd("../../data_original/CentralParkSoil")

library(dplyr)
library(stringr)
library(ape)

# S.H.: this script is adapted from https://github.com/jacobbien/trac-reproducible/blob/main/CentralParkSoil/original/Data%20Analysis%20Pipeline%20for%20Soil%20PH%20Dataset.R
# Currently it is run first to create the csv files in the CentralParkSoil folder

### OTU Table
Data <- read.csv('original/otu_table_wTax_40000_filt2.txt',sep='\t',row.names=1,quote='#',)
taxonomy <- Data[,dim(Data)[2]]
Tax <- data.frame('OTU'=rownames(Data),'taxonomy'=taxonomy)
Data <- Data[,1:(dim(Data)[2]-1)] # p x n each column is one observation

otus <- as.list(rownames(Data)) %>% 
  sapply(.,FUN = function(n) {
    str_replace(n,'\"','') %>% 
      str_replace(.,'\"','')}
    )
rownames(Data) <- otus
Tax[,1] <- otus # two columns: OTU name and full path

### Clean up Tax labels (S.H. added: remove white space, covert "unclassified" to "<level>__", and add a number for uniqueness)

# dummy counts for missing labels
taxa_counts <- rep(1, 7)
taxa_levels <- c('k', 'p', 'c', 'o', 'f', 'g', 's')

clean_label <- function(label_str) {
  label_split <- strsplit(label_str, "; ")[[1]]
  label_split_nchar <- as.vector((sapply(label_split, nchar)))
  idx_miss <- which(tolower(label_split) == "unclassified" | label_split_nchar == 3)
  if (length(idx_miss) >= 1) {
    for (ii in idx_miss) {
      label_split[ii] <- paste0(taxa_levels[ii], "__", taxa_counts[ii])
    }
    taxa_counts[min(idx_miss)] <<- taxa_counts[min(idx_miss)] + 1
  }
  return(paste0(label_split, collapse = ';'))
}

Tax$taxonomy <- gsub('"', '', Tax$taxonomy)
Tax$taxonomy <- as.character(lapply(Tax$taxonomy, clean_label))

rownames(Tax) <- Tax$OTU
Tax <- Tax %>%
  select(-OTU)

### Independent Variable
MAP <- read.csv('original/CP_map.txt',sep='\t',header = T)
X <- MAP$pH
names(X) <- MAP$X.SampleID
nms <- sapply(as.list(MAP$X.SampleID),toString)
nms <- sapply(as.list(nms),FUN=function(x) paste('X.',x,'.',sep=''))
Data <- Data[,nms]

### Phylogeny
tree <- read.tree('original/rep_set_aligned_gapfilt20_filt2.tre')
otus <- tree$tip.label
Data <- Data[tree$tip.label,]

### Save to csv (S.H.)
# clean up column names of Data
colnames(Data) <- as.character(lapply(colnames(Data), function(x) {
  x <- gsub("^X.", '', x)
  x <- gsub(".$", '', x)
  x
}))

all(colnames(Data) == names(X)) # check order of obs are the same
data_df <- as.data.frame(Data)
data_df[['otu']] <- rownames(data_df)
rownames(data_df) <- NULL
tax_df <- as.data.frame(Tax)
tax_df[['otu']] <- rownames(tax_df)
rownames(tax_df) <- NULL
# group the otus that belong to the same taxa
data_df <- data_df %>%
  dplyr::left_join(tax_df, by="otu") %>%
  dplyr::select(-otu)
data_X <- rowsum(data_df[,1:580], data_df$taxonomy)
data_y <- unname(X)
names(data_y) <- colnames(Data)

# write to the subfolder under data_original
write.csv(data_X, "soil_X.csv", row.names = TRUE)
write.csv(data_y, "soil_y.csv", row.names = TRUE)
