# %%
# load libs
# ^^^^^^
# %load_ext autoreload
# %autoreload 2

from matplotlib import rc
import sys  # nopep8
sys.path.insert(0, "../")  # nopep8

from classo import classo_problem
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from src.load_data import *
from src.kernels_jax import *
from src.cfi import *
from src.utils import *
import pandas as pd
import numpy as np
from os.path import join, dirname, abspath
import pickle
from datetime import date

seed_num = 2022
today = date.today()

rc('font', **{'family': 'tex-gyre-termes', 'size': 6.5})
rc('text', usetex=True)
rc('text.latex',
   preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')

# plt.cm.tab10(range(10)), plt.cm.tab20(range(20)), plt.cm.tab20b(range(20)),
colors_all = np.vstack([plt.cm.tab20c(range(20)), plt.cm.tab20b(range(20))])
colors_all = np.unique(colors_all, axis=0)
print(colors_all.shape)

# %%
# load data
# ^^^^^^

y_labels, X_refseq_taxa, X_refseq_otu, X_meta = load_data(
    "qin2014", data_path="../data/MLRepo/qin2014")
y = y_labels.Var.to_numpy()
y = np.array([1 if y_i == 'Healthy' else -1 for y_i in y])

# aggregate to species
X_refseq_taxa_parsed = parse_taxa_spec(X_refseq_taxa)
X_refseq_taxa = group_taxa(X_refseq_taxa_parsed, levels=["kingdom", "phylum", "class", "order",
                                                         "family", "genus", "species"])

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

# %%
# prepare data
# ^^^^^^

X_df = pd.DataFrame(X_count.T, index=comp_lbl, columns=[
                    'SUB_'+str(k) for k in range(X_count.shape[0])])
label = comp_lbl

# nanual progressive filtering
beta0 = -10
beta1 = 1
X_df, cols_keep = manual_filter(X_df, beta0, beta1, plot=True)
label = label[cols_keep]

print(X_df.shape)
print(label.shape)

# %%
# median heruisic for RBF and Aitchison-RBF
# ^^^^^^

X_comp = X_df.to_numpy().astype('float').T
X_comp /= X_comp.sum(axis=1)[:, None]

# for rbf
K = squared_euclidean_distances(X_comp, X_comp)
k_triu = K[np.triu_indices(n=K.shape[0], k=1, m=K.shape[1])]
g1 = 1.0/np.median(k_triu)
print(g1)

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

param_grid_svm = dict(C=[10**x for x in [-3, -2, -1, 0, 1, 2, 3]])
print(param_grid_svm)

kernel_params_dict = default_kernel_params_grid(g1, g2)
kmat_with_params = get_kmat_with_params(kernel_params_dict)
mod_with_params = kmat_with_params
print(f"number of kernels: {len(mod_with_params)}")

# %%
# fit and select top models in each group
# ^^^^^^

X_all = X_df.to_numpy().astype('float').T
train_scores_all, test_scores_all, selected_params_all = run_experiments(X_all, y, mod_with_params, param_grid_svm, None, None,
                                                                         center_kmat=False, fac_grid=None, n_fold_outer=10, n_fold_inner=5,
                                                                         type='classification', scoring='accuracy', kernel_estimator='svm',
                                                                         n_jobs=-1, random_state=seed_num, verbose=0)
best_models = top_models_in_each_group(
    mod_with_params, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
best_models.to_csv(
    f"cirrhotic_unweighted_best_models_{today.strftime('%b-%d-%Y')}.csv", index=False)


# %%
# refit the best kernel
# ^^^^^^

X_all /= X_all.sum(axis=1)[:, None]
model_selected = best_models.iloc[0]
print(model_selected)
pred_fun, gscv = refit_best_model(X_all, y, 'SVC', param_grid_svm, model_selected,
                                  'accuracy', center_kmat=False, n_fold=5, n_jobs=6, verbose=0)
print(gscv.best_estimator_)

# %%
# unweighted CFI and CPD values
# ^^^^^^^

df = df_ke_dual_mat(X_all, X_all, gscv.best_estimator_.dual_coef_[
                    0], gscv.best_estimator_.support_, kernel_args_str_to_k_fun(model_selected.estimator_key))
cirrhotic_cfi_vals = get_perturbation_cfi_index(X_all, df)
cirrhotic_cpd_vals = get_cfi(X_all, X_all, pred_fun)

np.save("cirrhotic_unweighted_cfi_df.npy", df)
np.save("cirrhotic_unweighted_cfi_vals.npy", cirrhotic_cfi_vals)
np.save("cirrhotic_unweighted_cpd_vals.npy", cirrhotic_cpd_vals)
