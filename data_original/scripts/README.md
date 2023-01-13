The scripts here are mainly for harmonizing data stored in specific format such as `*.RData`, `*.rds`, `*.biom`, or `*.tre`, which we obtained from the [trac-reproducible](https://github.com/jacobbien/trac-reproducible) repository. 

The output of these scripts should be a matrix $X$ of dimension `n_features x n_observations` and a vector $y$ of length `n_observations`. The $X$ matrix may have row names (if available) being the full phylogenetic path, e.g., "k__Bacteria;p__Verrucomicrobia;c__[Pedosphaerae];o__[Pedosphaerales];f__1;g__1;s__1" and the names should be unique.

The other than the `scripts` folder, the `data_original` folder should contain the following folders, each should contain the files in the corresponding links ("*name*/original" means having a subfolder "original" under "*name*"):
- `AmericanGut/original`: https://github.com/jacobbien/trac-reproducible/tree/main/AmericanGut/original
- `CAMP`: https://microbiomedb.org/mbio/app/downloads/release-28/fe8cd04882903402ee64a9dc3e23bcbdf604040e/
- `CentralParkSoil/original`: https://github.com/jacobbien/trac-reproducible/tree/main/CentralParkSoil/original
- `GenomeMed`: https://github.com/SchlossLab/Baxter_glne007Modeling_GenomeMed_2015/tree/master/data
- `Marine/original`: https://github.com/jacobbien/trac-reproducible/tree/main/Marine/original
- `MLRepo`: http://metagenome.cs.umn.edu/public/MLRepo/datasets.tar.gz
- `RMP`: upon request to Shimeng Huang (originally requested from the authors of https://www.nature.com/articles/nature24460) 
- `sCD14_HIV/original`: https://github.com/jacobbien/trac-reproducible/tree/main/sCD14_HIV/original
