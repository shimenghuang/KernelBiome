library(dplyr)
library(stringr)
library(ape)

### OTU Table
Data <- read.csv('otu_table_wTax_40000_filt2.txt',sep='\t',row.names=1,quote='#',)
taxonomy <- Data[,dim(Data)[2]]
Tax <- data.frame('OTU'=rownames(Data),'taxonomy'=taxonomy)
Data <- Data[,1:(dim(Data)[2]-1)]

otus <- as.list(rownames(Data)) %>% sapply(.,FUN = function(n)  str_replace(n,'\"','') %>% str_replace(.,'\"',''))
rownames(Data) <- otus
Tax[,1] <- otus

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

# clean_label <- function(label_str) {
#   label_split <- strsplit(label_str, "; ")[[1]]
#   taxa_levels <- c('k', 'p', 'c', 'o', 'f', 'g', 's')
#   for (ii in 1:length(label_split)) {
#     if (!grepl('[kpcofgs]__', label_split[ii])) {
#     # if (label_split[ii] == "unclassified" | nchar(label_split[ii]) == 3) {
#       label_split[ii] <- paste0(taxa_levels[ii], "__", taxa_counts[ii])
#       taxa_counts[1:ii] <<- taxa_counts[1:ii] + 1 # increase counts for levels above too
#     }
#   }
#   return(paste0(label_split, collapse = ';'))
# }

Tax$taxonomy <- gsub('"', '', Tax$taxonomy)
Tax$taxonomy <- as.character(lapply(Tax$taxonomy, clean_label))
# Tax <- Tax %>% 
#   group_by(taxonomy) %>% 
#   mutate(num = 1:n()) %>%
#   ungroup() %>%
#   mutate(taxonomy = paste0(taxonomy, num)) %>%
#   select(-num) %>%
#   as.data.frame()
## make leave names unique
# leaf_no_name <- grepl("s__$", Tax$taxonomy)
# append_num <- 1:sum(leaf_no_name)
# Tax$taxonomy[leaf_no_name] <- paste0(Tax$taxonomy[leaf_no_name], append_num)
# Tax$taxonomy <- paste0(Tax$taxonomy, paste0("dummy", 1:length(Tax$taxonomy)))

rownames(Tax) <- Tax$OTU
Tax <- Tax %>%
  select(-OTU)

### Independent Variable
MAP <- read.csv('CP_map.txt',sep='\t',header = T)
X <- MAP$pH
names(X) <- MAP$X.SampleID
nms <- sapply(as.list(MAP$X.SampleID),toString)
nms <- sapply(as.list(nms),FUN=function(x) paste('X.',x,'.',sep=''))
Data <- Data[,nms]

### Phylogeny
tree <- read.tree('rep_set_aligned_gapfilt20_filt2.tre')
otus <- tree$tip.label
Data <- Data[tree$tip.label,]

### Save to csv (S.H.)
# clean up column names of Data
colnames(Data) <- as.character(lapply(colnames(Data), function(x) {
  x <- gsub("^X.", '', x)
  x <- gsub(".$", '', x)
  x
}))
write.csv(Data, "soil_X.csv")
write.csv(X, "soil_y.csv")
write.csv(Tax, "soil_tax.csv")
