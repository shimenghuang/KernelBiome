"""
Simple Collection of load functions for the various datasets

For taxa data (usage example see load_data.ipynb):
   load all data (`load_data`)
   => aggregate taxa if desired (`parse_taxa_spec_MLRepo` and `group_taxa_MLRepo`)
   => filtering if desired (`progressive_filter` or `RF_filter`)
   => renormalize and transpose into proper shape
   => X of shape (n_sample, n_comp), y of shape (n_sample,), and component labels of shape (n_sample,) all of type np.array

"""

from audioop import avg
import pandas as pd
import os
import numpy as np
from scipy import rand
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import rdata
import matplotlib.pyplot as plt


def load_data(ds_name, data_path=None):
    """
    General function for data loading

    Parameters
    ----------
    ds_name: str
        name of dataset, one of "qin2014" (data obtained from MLRepo) and "soil" (data obtained from https://github.com/elies-ramon/kernInt/blob/master/data/soil.rda).
    data_path: str
        if ds_name == "qin2014", this path should lead to the folder containing "otutable.txt", "taxatable.txt", and "mapping-orig.txt";
        if ds_name == "soil", this path should lead to a folder containing "otu_table_paper.csv", "tax_table_phyla.csv", and "ph_value.csv".

    Returns
    -------
    y_labels: pd.DataFrame
        index is the subject id's and one column named 'Var'.
    X_taxa: pd.DataFrame
        index is the full taxa names concatenated by ';', columns are the subject ID's.
        Note: for qin2014, this is X_refseq_taxa, and X_refseq_otu has different row names, the otu id's.
    X_meta: pd.DataFrame
        orignal metadata, in case ndeeded.
    """

    if ds_name == "qin2014":

        X_refseq_otu = pd.read_csv(os.path.join(
            data_path, "otutable.txt"), sep="\t")
        X_refseq_otu.set_index('#OTU ID', inplace=True)
        X_refseq_taxa = pd.read_csv(os.path.join(
            data_path, "taxatable.txt"), sep="\t")
        X_refseq_taxa.set_index('#OTU ID', inplace=True)
        X_meta = pd.read_csv(os.path.join(
            data_path, "mapping-orig.txt"), sep="\t")

        y_labels = pd.read_csv(os.path.join(
            data_path, "task-healthy-cirrhosis.txt"), sep="\t")
        y_labels.set_index('#SampleID', inplace=True)

        return y_labels, X_refseq_taxa, X_refseq_otu, X_meta

    if ds_name == "soil":

        parsed = rdata.parser.parse_file(os.path.join(
            data_path, "soil.rda"))
        converted = rdata.conversion.convert(parsed)

        X_meta = converted['soil']['metadata'].sort_index()
        taxonomy = converted['soil']['taxonomy']
        taxo_concat = taxonomy[:].agg(';'.join, axis=1).values
        taxo_df = pd.DataFrame(data=taxo_concat, index=taxonomy.index.astype(
            'str'), columns=['#OTU ID'])
        taxo_df.index.names = ['taxa_id']
        abund = converted['soil']['abund'].to_pandas(
        ).sort_index().reset_index(drop=True).T
        abund.index.names = ['taxa_id']
        abund.rename(columns=lambda x: X_meta.index.values[x], inplace=True)
        X_taxa = pd.merge(taxo_df, abund, left_index=True, right_index=True).reset_index(
            drop=True).set_index('#OTU ID')
        y_labels = pd.DataFrame(
            data=X_meta['ph'].values, index=X_meta.index, columns=['Var'])

        return y_labels, X_taxa, X_meta


def parse_taxa_spec(X_taxa):
    """
    There are 8 levels in total: kingdom; phylum; class; order; family; genus; species; strain

    X_taxa: pd.DataFrame
        columns are subject labels and index contains taxa names in the format of e.g. "k__Archaea;p__Crenarchaeota;c__Thaumarchaeota;o__Cenarchaeales;f__SAGMA-X;g__2517"
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


def group_taxa(X_taxa_split, levels=['kingdom', 'phylum']):
    """
    X_taxa_split: return value from `parse_taxa_spec_MLRepo`
    level: a list containing upto all levels kingdom; phylum; class; order; family; genus; species; strain
    """
    X_taxa = X_taxa_split.groupby(levels).sum()
    X_taxa.index = X_taxa.index.map(';'.join)
    return X_taxa


def progressive_filter(X_df):
    """
    Progressive filtering as described in https://www.biorxiv.org/content/10.1101/2022.01.03.474763v1

    Parameters
    ----------
    X_df: pd.DataFrame
      rows are the OTU/Taxa labels and columns are subjects.

    Returns
    -------
    X_keep: pd.DataFrame
        filtered dataset
    """

    X_np = X_df.to_numpy().T.astype('float64')
    # prevalence: proportion of samples that have taxa j
    prevalence = 1 - (X_np == 0).sum(axis=0) / X_np.shape[0]
    # average abundance: relative abundance averaged across samples
    avg_abundance = np.mean(X_np / X_np.sum(axis=1)[:, None], axis=0)
    beta0 = (np.median(prevalence) - np.quantile(prevalence, 0.25)) / (
        np.quantile(avg_abundance, 0.25) - np.median(avg_abundance))
    beta1 = np.quantile(prevalence, 0.25) - beta0 * np.median(avg_abundance)
    # beta0 = (np.median(prevalence) - np.quantile(prevalence, prev_qt)) / (
    #     np.quantile(avg_abundance, abun_qt) - np.median(avg_abundance))
    # beta1 = np.quantile(prevalence, prev_qt) - beta0 * np.median(avg_abundance)
    taxa_keep = np.where(prevalence >= avg_abundance * beta0 + beta1)[0]

    X_keep = X_df.iloc[taxa_keep, :]
    # Warning: this formatting could be very MLRepo specific
    # X_keep.index = X_keep.index.map(';'.join)

    # remove the NaN OTU ID
    taxa_keep = taxa_keep[pd.notna(X_keep.index)]
    X_keep = X_keep.loc[pd.notna(X_keep.index)]

    return X_keep, taxa_keep


def manual_filter(X_df, beta0=-75, beta1=0.9, plot=False):
    """
    Progressive filtering as described in https://www.biorxiv.org/content/10.1101/2022.01.03.474763v1

    Parameters
    ----------
    X_df: pd.DataFrame
      rows are the OTU/Taxa labels and columns are subjects.

    Returns
    -------
    X_keep: pd.DataFrame
        filtered dataset
    """

    X_np = X_df.to_numpy().T.astype('float64')
    # prevalence: proportion of samples that have taxa j
    prevalence = 1 - (X_np == 0).sum(axis=0) / X_np.shape[0]
    # average abundance: relative abundance averaged across samples
    avg_abundance = np.mean(X_np / X_np.sum(axis=1)[:, None], axis=0)
    taxa_keep = np.where(prevalence >= (avg_abundance * beta0 + beta1))[0]

    X_keep = X_df.iloc[taxa_keep, :]
    # remove the NaN OTU ID
    taxa_keep = taxa_keep[pd.notna(X_keep.index)]
    X_keep = X_keep.loc[pd.notna(X_keep.index)]

    if plot:
        plt.scatter(avg_abundance, prevalence, s=1)
        plt.axline((0, beta1), slope=beta0)
        plt.xlabel('averge abundance')
        plt.ylabel('prevalence')
        plt.show()

    return X_keep, taxa_keep


def prevalence_filter(X_df, threshold=0.75):
    X_np = X_df.to_numpy().T.astype('float64')
    # prevalence: proportion of samples that have taxa j
    prevalence = 1 - (X_np == 0).sum(axis=0) / X_np.shape[0]
    taxa_keep = prevalence >= threshold
    X_keep = X_df.iloc[taxa_keep, :]
    return X_keep, taxa_keep

# def progressive_filtering(X):

#     prevalence = 1 - (X == 0).sum(axis=0) / X.shape[0]
#     avg_abundance = (X / np.sum(X, axis=1, keepdims=True)
#                      ).sum(axis=0) / X.shape[0]
#     beta0 = (np.median(prevalence) - np.quantile(prevalence, 0.25)) / (
#         np.quantile(avg_abundance, 0.25) - np.median(avg_abundance))
#     beta1 = np.quantile(prevalence, 0.25) - beta0 * np.median(avg_abundance)
#     taxa_keep = prevalence >= avg_abundance * beta0 + beta1
#     X_keep = X[:, taxa_keep]
#     # Warning: this formatting could be very MLRepo specific
#     #X_keep.index = X_keep.index.map(';'.join)

#     # remove the NaN OTU ID
#     #X_keep = X_keep.loc[pd.notna(X_keep.index)]

#     return X_keep, taxa_keep


def RF_filter(X, y, n_top_features=50, type='clf', random_state=None):
    """
    apply random forest to select top features and trim down the space

    Parameters
    ----------
    X: np.ndarray
        compositional vector
    y: np.ndarray
        labels
    n_top_features: int=50
        number of top features
    type:
        one of 'clf' or 'reg'

    Returns
    -------
    """
    assert(type == 'clf' or type == 'reg')
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=random_state) if type == 'clf' else RandomForestRegressor(
        n_estimators=100, max_depth=2, random_state=random_state)
    clf.fit(X, y)
    cols_keep = np.argsort(-clf.feature_importances_)[:n_top_features]

    X_red = X[:, cols_keep]

    return X_red, cols_keep

# def RF_filter(X, y, n_top_features=50, is_binary=True):
#     """
#     apply random forest to select top features and trim down the space

#     Parameters
#     ----------
#     X: np.ndarray
#         compositional vector
#     y: np.ndarray
#         labels
#     n_top_features: int=50
#         number of top features

#     Returns
#     -------
#     """

#     if is_binary:
#         model = RandomForestClassifier(n_estimators=100, max_depth=2)
#     else:
#         model = RandomForestRegressor(n_estimators=100, max_depth=2)

#     model.fit(X, y)
#     cols_keep_comp = np.argsort(-model.feature_importances_)[:n_top_features]

#     print(sorted(cols_keep_comp))
#     cols_keep = cols_keep_comp

#     X_red = X[:, cols_keep]

#     return X_red, cols_keep
