# %%
# load libs
# ^^^^^^

import sys
from os.path import join
import pandas as pd
import numpy as np
from load_data_helper import *


# %%
# load data
# ^^^^^^

def load_agg_shuffle(data_path, seed_num=2022):

    # load original data
    X_orig = pd.read_csv(join(data_path, "soil_X.csv"))  #
    X_orig.rename(columns={"Unnamed: 0": "OTU"}, inplace=True)
    X_orig.set_index('OTU', inplace=True)
    taxa_all = pd.read_csv(join(data_path, "soil_tax.csv"))
    taxa_all.rename(columns={"Unnamed: 0": "OTU"}, inplace=True)
    taxa_all.set_index('OTU', inplace=True)
    X_df = X_orig.join(taxa_all, on='OTU',
                       how='left').reset_index().drop('OTU', axis=1)
    X_df.set_index('taxonomy', inplace=True)
    y = pd.read_csv(join(data_path, "soil_y.csv"))
    y.rename(columns={"Unnamed: 0": "SUBJ", "x": "PH"}, inplace=True)
    y.set_index('SUBJ', inplace=True)
    # sort y according to columns of X_df
    y = y.loc[X_df.columns.to_numpy()].PH.to_numpy()
    otu_label = X_df.index.to_numpy()
    x = X_df.T.to_numpy()

    print(x.shape)
    print(y.shape)
    print(otu_label.shape)

    # shuffle once only
    rng = np.random.default_rng(seed_num)
    idx_shuffle = rng.choice(range(x.shape[0]), x.shape[0], replace=False)

    x = x[idx_shuffle]
    y = y[idx_shuffle]

    return x, y, otu_label

# %%
# prepare data
# ^^^^^^


def filter_prep(x, otu_label):

    X_df = pd.DataFrame(x.T, index=otu_label, columns=[
        'SUB_'+str(k) for k in range(x.shape[0])])
    label = otu_label

    # Opt 2) do manual progressive filtering
    beta0 = -100
    beta1 = 1.03
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

# %%
