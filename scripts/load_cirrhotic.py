# %%
# load libs
# ^^^^^^

import sys
import pandas as pd
import numpy as np
from load_data_helper import *


# %%
# load data
# ^^^^^^

def load_agg_shuffle(data_path, seed_num=2022):

    y_labels, X_refseq_taxa, X_refseq_otu, X_meta = load_data(
        "qin2014", data_path=data_path)
    y = y_labels.Var.to_numpy()
    y = np.array([1 if y_i == 'Healthy' else -1 for y_i in y])

    # aggregate to species
    X_refseq_taxa_parsed = parse_taxa_spec(X_refseq_taxa)
    X_refseq_taxa = group_taxa(X_refseq_taxa_parsed, levels=["kingdom", "phylum", "class", "order",
                                                             "family", "genus", "species"])

    # comp_lbl = X_refseq_taxa['comp_lbl']
    X_count = X_refseq_taxa.T.to_numpy()
    comp_lbl = X_refseq_taxa.index.to_numpy()
    print(X_count.shape)
    print(y.shape)
    print(comp_lbl.shape)

    # shuffle once only
    rng = np.random.default_rng(seed_num)
    idx_shuffle = rng.choice(
        range(X_count.shape[0]), X_count.shape[0], replace=False)
    X_count = X_count[idx_shuffle]
    y = y[idx_shuffle]

    return X_count, y, comp_lbl

# %%
# prepare data
# ^^^^^^


def filter_prep(X_count, comp_lbl):

    X_df = pd.DataFrame(X_count.T, index=comp_lbl, columns=[
                        'SUB_'+str(k) for k in range(X_count.shape[0])])
    label = comp_lbl.copy()

    # manual progressive filtering
    beta0 = -10
    beta1 = 1
    X_df, cols_keep = manual_filter(X_df, beta0, beta1, plot=True)
    label = label[cols_keep]

    print(X_df.shape)
    print(label.shape)

    return X_df, label

# %%
# main
# ^^^^^^^


def main(data_path, seed_num=2022):
    X_count, y, comp_lbl = load_agg_shuffle(data_path, seed_num)
    X_df, label = filter_prep(X_count, comp_lbl)
    return X_df, y, label


if __name__ == "__main__":
    main(sys.argv[1])
