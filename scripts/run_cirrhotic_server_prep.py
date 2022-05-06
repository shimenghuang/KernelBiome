# %%
# load libs
# ^^^^^^

from classo import classo_problem
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from src.load_data import *
from src.kernels_jax import *
from src.cfi import *
from src.utils import *
import pandas as pd
import numpy as np


# %%
# load data
# ^^^^^^
if on_computerome:
    data_path = "kernelbiome/data/MLRepo/qin2014"
else:
    data_path = "/Users/hrt620/Documents/projects/kernelbiome_proj/kernelbiome/data/MLRepo/qin2014"

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
# TODO: probably we do have to change that for the weighted kernels -> there have to be less of those
beta0 = -10
beta1 = 1
X_df, cols_keep = manual_filter(X_df, beta0, beta1, plot=True)
label = label[cols_keep]

print(X_df.shape)
print(label.shape)

# data to be used for classo (with added pseudo count)
pseudo_count = 1  # Note: pseudo count seems to matter
X = np.log(pseudo_count + X_df.to_numpy().astype('float').T)

# %%
# median heruisic for RBF and Aitchison-RBF
# ^^^^^^

X_comp = X_df.to_numpy().astype('float').T
X_comp /= X_comp.sum(axis=1)[:, None]

# for rbf
K = squared_euclidean_distances(X_comp, X_comp)
k_triu = K[np.triu_indices(n=K.shape[0], k=1, m=K.shape[1])]
g1 = 1.0/np.median(k_triu)
# g2 = 1.0/np.median(np.sqrt(k_triu))**2
print(g1)
# print(g2)

# for aitchison-rbf
Xc = X_comp + 1e-5
gm_Xc = gmean(Xc, axis=1)
clr_Xc = np.log(Xc/gm_Xc[:, None])
K = squared_euclidean_distances(clr_Xc, clr_Xc)
k_triu = K[np.triu_indices(n=K.shape[0], k=1, m=K.shape[1])]
g2 = 1.0/np.median(k_triu)
print(g2)

# %%
# setup for KB estimator
# ^^^^^^


def my_kernel_params_grid():
    kernel_params_dict = {
        'linear': None,
        # 'rbf': {'g': np.logspace(-2, 2, 5)},
        'rbf': {'g': get_rbf_bandwidth(g1)},
        # Note: hilbert1, only valid when a >= 1, 0.5 <= b <= a
        'generalized-js': {'a': [1, 10, np.inf], 'b': [0.5, 1, 10, np.inf]},
        # Note: hilbert2, only valid when a >= 1 b <= -1 and not both a, b are inf
        'hilbertian': {'a': [1, 10, np.inf], 'b': [-1, -10, -np.inf]},
        'aitchison': {'c': np.logspace(-7, -3, 5)},
        # 'aitchison-rbf': {'c': np.logspace(-7, -3, 5), 'g': np.logspace(-2, 2, 5)},
        'aitchison-rbf': {'c': np.logspace(-7, -3, 5), 'g': get_rbf_bandwidth(g2)},
        # TODO: experimental range from the data so that hd values do not blow up
        # 'heat-diffusion': {'t': np.linspace(0.075, 0.1, 5)}
        'heat-diffusion': {'t': np.linspace(0.9, 1.1, 5)*0.25/np.pi}
    }
    return kernel_params_dict


param_grid_lr = dict(C=[10**x for x in [-3, -2, -1, 0, 1, 2, 3]])
print(param_grid_lr)
param_grid_svm = dict(C=[10**x for x in [-3, -2, -1, 0, 1, 2, 3]])
print(param_grid_svm)
param_grid_kr = dict(alpha=list(np.logspace(-7, 1, 5, base=2)))
print(param_grid_kr)
param_grid_rf = dict(n_estimators=[10, 20, 50, 100, 250, 500])
print(param_grid_rf)
param_grid_baseline = dict(strategy=["mean", "median"])
print(param_grid_baseline)

kernel_params_dict = my_kernel_params_grid()
kmat_with_params = get_kmat_with_params(kernel_params_dict)
# add RF and baseline
mod_with_params = kmat_with_params

# %%
# setup for WKB estimator
# ^^^^^^


def my_weighted_kernel_params_grid(w_unifrac):
    kernel_params_dict = {
        'linear_weighted': {'w': w_unifrac},
        # 'rbf_weighted': {'g': np.logspace(-2, 2, 5), "w": w_unifrac},
        'rbf_weighted': {'g': [np.sqrt(g1), g1, g1**2, g1**3, g1**4], "w": w_unifrac},
        # Note: only valid
        'generalized-js_weighted': {'a': [1, 10, np.inf], 'b': [0.5, 1, 10, np.inf], "w": w_unifrac},
        # Note: only valid when a >= 1 b <= -1 and not both a, b are inf
        'hilbertian_weighted': {'a': [1, 10, np.inf], 'b': [-1, -10, -np.inf], "w": w_unifrac},
        'aitchison_weighted': {'c': np.logspace(-7, -3, 5), 'w': w_unifrac},
        # 'aitchison-rbf_weighted': {'c': np.logspace(-7, -3, 5), 'g': np.logspace(-2, 2, 5), 'w': w_unifrac},
        'aitchison-rbf_weighted': {'c': np.logspace(-7, -3, 5), 'g': [1.0/g2, g2**(-0.5), np.sqrt(g2), g2, g2**2], 'w': w_unifrac},
        # 'heat-diffusion_weighted': {'t': np.linspace(0.075, 0.1, 5), 'w': w_unifrac}
        'heat-diffusion_weighted': {'t': np.linspace(0.9, 1.1, 5)*0.25/np.pi, 'w': w_unifrac}
    }
    return kernel_params_dict


# compute unifrac weight
levels = ["kingdom", "phylum", "class",
          "order", "family", "genus", "species"]
delim = ";"
w_unifrac_orig = np.load(
    join(file_dir, "w_unifrac_cirrhotic_opt1.npy"), allow_pickle=True)
w_unifrac_psd = np.load(
    join(file_dir, "w_unifrac_cirrhotic_opt3.npy"), allow_pickle=True)

# original unifrac weights
weighted_kernel_params_dict_orig = my_weighted_kernel_params_grid(
    w_unifrac_orig)
weighted_kmat_with_params_orig = get_weighted_kmat_with_params(
    weighted_kernel_params_dict_orig, w_unifrac=w_unifrac_orig)
weighted_mod_with_params_orig = weighted_kmat_with_params_orig

# psd-ed unifrac weights
weighted_kernel_params_dict_psd = my_weighted_kernel_params_grid(
    w_unifrac_psd)
weighted_kmat_with_params_psd = get_weighted_kmat_with_params(
    weighted_kernel_params_dict_psd, w_unifrac=w_unifrac_psd)
weighted_mod_with_params_psd = weighted_kmat_with_params_psd

# %%
