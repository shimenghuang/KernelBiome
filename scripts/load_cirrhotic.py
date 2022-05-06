# %%
# load libs
# ^^^^^^

import sys
from src.load_data import *
from src.kernels_jax import *
from src.cfi import *
from src.utils import *
import pandas as pd
import numpy as np

# %%
# load data
# ^^^^^^

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
rng = np.random.default_rng(2022)
idx_shuffle = rng.choice(
    range(X_count.shape[0]), X_count.shape[0], replace=False)
X_count = X_count[idx_shuffle]
y = y[idx_shuffle]

# name of biom and tree file
biom_file = "cirrhotic_1sample.biom"
tree_file = "cirrhotic_tree.tre"

# %%
# prepare data
# ^^^^^^

X_df = pd.DataFrame(X_count.T, index=comp_lbl, columns=[
                    'SUB_'+str(k) for k in range(X_count.shape[0])])
label = comp_lbl

# Opt 2) do manual progressive filtering
beta0 = -10
beta1 = 1
X_df, cols_keep = manual_filter(X_df, beta0, beta1, plot=True)
label = label[cols_keep]

print(X_df.shape)
print(label.shape)

# %%
