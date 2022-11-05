The scripts here are mainly for harmonizing data stored in specific format such as `*.RData`, `*.rds`, `*.biom`, or `*.tre`, which we obtained from the [trac-reproducible](https://github.com/jacobbien/trac-reproducible) repository. 

The output of these scripts should be a matrix $X$ of dimension `n_features x n_observations` and a vector $y$ of length `n_observations`. The $X$ matrix may have row names (if available) being the full phylogenetic path, e.g., "k__Bacteria;p__Verrucomicrobia;c__[Pedosphaerae];o__[Pedosphaerales];f__1;g__1;s__1" and the names should be unique.




