import numpy as np
import pandas as pd


def filter_taxa(X_pn, threshold, verbose=True):

    X_np = X_pn.to_numpy().T.astype('float64')
    # median_count: median count if species exists
    median_count = np.zeros(X_np.shape[1])
    for j in range(X_np.shape[1]):
        if np.sum(X_np[:, j] != 0) != 0:
            median_count[j] = np.median(X_np[X_np[:, j] != 0, j])
    # prevalence: proportion of samples that have taxa j
    prevalence = 1 - (X_np == 0).sum(axis=0) / X_np.shape[0]
    # select taxa with given threshold
    taxa_keep = np.where((prevalence >= threshold[0]) &
                         (median_count >= threshold[1]))[0]
    X_df = X_pn.iloc[taxa_keep, :]

    if verbose:
        print(X_df.shape)

    return X_df


def parse_taxa_spec(X_taxa):
    """Create an intermediate dataframe when aggregating levels.

    There are 8 levels in total: kingdom; phylum; class; order;
    family; genus; species; strain

    X_taxa: pd.DataFrame columns are subject labels and (row) index
        contains taxa names in the format of
        e.g. "k__Archaea;p__Crenarchaeota;c__Thaumarchaeota;o__Cenarchaeales;
        f__SAGMA-X;g__2517"

    """
    X_taxa_loc = X_taxa.reset_index()
    X_taxa_loc.rename(columns={X_taxa_loc.columns[0]: 'index'}, inplace=True)
    X_taxa_split = pd.concat(
        [X_taxa_loc, X_taxa_loc['index'].str.split(';', expand=True)], axis=1)
    X_taxa_split.rename(columns={0: 'kingdom',
                                 1: 'phylum',
                                 2: 'class',
                                 3: 'order',
                                 4: 'family',
                                 5: 'genus',
                                 6: 'species',
                                 7: 'strain'},
                        inplace=True)
    return X_taxa_split


def group_taxa(X_df, grp_to='genus'):
    """X_df: a pd.DataFrame where columns are subject labels and (row)
    index contains taxa names in the format of

    e.g. "k__Archaea;p__Crenarchaeota;c__Thaumarchaeota;o__Cenarchaeales;
    f__SAGMA-X;g__2517"
    level: a list containing upto all levels kingdom; phylum; class;
    order; family; genus; species; strain

    """
    all_levels = ["kingdom", "phylum", "class",
                  "order", "family", "genus", "species", "strain"]
    idx_grp_to = all_levels.index(grp_to)
    levels = all_levels[:(idx_grp_to+1)]
    X_df_split = parse_taxa_spec(X_df)
    X_gpd = X_df_split.groupby(levels).sum(numeric_only=True)
    X_gpd.index = X_gpd.index.map(';'.join)
    return X_gpd
