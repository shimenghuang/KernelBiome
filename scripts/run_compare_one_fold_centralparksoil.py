# %%
# load libs
# ^^^^^^

on_computerome = True  # nopep8

import sys  # nopep8
if on_computerome:  # nopep8
    sys.path.insert(0, "./KernelBiome/")  # nopep8
else:  # nopep8
    sys.path.insert(0, "../")  # nopep8

from os.path import join
import numpy as np
from src.helpers_jax import *
from src.utils import *
import load_centralparksoil
import compare_inner_fold_centralparksoil

fold_idx = int(sys.argv[1])
print(f"fold: {fold_idx}")
seed_num = 2022

# %%
# set up paths
# ^^^^^^

# input and output paths
if on_computerome:
    data_path = "KernelBiome/data/MLRepo/qin2014"
    output_path = "KernelBiome/scripts/output"
else:
    data_path = "/Users/hrt620/Documents/projects/kernelbiome_proj/kernelbiome_clean/data/MLRepo/qin2014"
    output_path = "/Users/hrt620/Documents/projects/kernelbiome_proj/kernelbiome_clean/scripts/output"

# intermediate results
do_save = True
do_save_file = join(
    output_path, f"centralparksoil_res_kb_fold_{fold_idx}.pickle")
do_save_file_weighted_ma = join(
    output_path, f"centralparksoil_res_MA_kb_fold_{fold_idx}.pickle")
do_save_file_weighted_mb = join(
    output_path, f"centralparksoil_res_MB_kb_fold_{fold_idx}.pickle")

# %%
# call prep script
# ^^^^^^

# data to be used for most of the methods
X_df, y, label = load_centralparksoil.main(
    data_path=data_path, seed_num=seed_num)

# data to be used for classo (with added pseudo count)
pseudo_count = 1
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
# setup hyperparameter search grid
# ^^^^^^

param_grid_lasso = dict(alpha=[10**x for x in [-5, -4, -3, -2, -1]])
print(param_grid_lasso)
param_grid_svm = dict(C=[10**x for x in [-3, -2, -1, 0, 1, 2, 3]])
print(param_grid_svm)
param_grid_kr = dict(alpha=list(np.logspace(-7, 1, 5, base=2)))
print(param_grid_kr)
param_grid_rf = dict(n_estimators=[10, 20, 50, 100, 250, 500])
print(param_grid_rf)
param_grid_baseline = dict(strategy=["mean", "median"])
print(param_grid_baseline)

kernel_params_dict = default_kernel_params_grid()
kmat_with_params = get_kmat_with_params(kernel_params_dict)
mod_with_params = kmat_with_params

# %%
# setup for WKB estimator
# ^^^^^^

# UniFrac weight with MA
w_unifrac_ma = np.load(
    join(output_path, "centralparksoil_w_unifrac_MA.npy"), allow_pickle=True)
weighted_kernel_params_dict_ma = default_weighted_kernel_params_grid(
    w_unifrac_ma, g1, g2)
weighted_kmat_with_params_ma = get_weighted_kmat_with_params(
    weighted_kernel_params_dict_ma, w_unifrac=w_unifrac_ma)

# UniFrac weight with MB
w_unifrac_mb = np.load(
    join(output_path, "centralparksoil_w_unifrac_MB.npy"), allow_pickle=True)
weighted_kernel_params_dict_mb = default_weighted_kernel_params_grid(
    w_unifrac_mb, g1, g2)
weighted_kmat_with_params_mb = get_weighted_kmat_with_params(
    weighted_kernel_params_dict_mb, w_unifrac=w_unifrac_mb)


# original unifrac weights
weighted_kernel_params_dict_orig = default_weighted_kernel_params_grid(
    w_unifrac_ma)
weighted_kmat_with_params_orig = get_weighted_kmat_with_params(
    weighted_kernel_params_dict_orig, w_unifrac=w_unifrac_ma)
weighted_mod_with_params_orig = weighted_kmat_with_params_orig

# psd-ed unifrac weights
weighted_kernel_params_dict_psd = default_weighted_kernel_params_grid(
    w_unifrac_mb)
weighted_kmat_with_params_psd = get_weighted_kmat_with_params(
    weighted_kernel_params_dict_psd, w_unifrac=w_unifrac_mb)
weighted_mod_with_params_psd = weighted_kmat_with_params_psd

# %%
# load CV indices
# ^^^^^^

n_compare = 50
comparison_cv_idx = np.load(
    join(output_path, f"output/centralparksoil_compare_{n_compare}cv_idx.npz"), allow_pickle=True)

# %%
# run a comparison CV fold
# ^^^^^^

tr = comparison_cv_idx['tr_list'][fold_idx]
te = comparison_cv_idx['te_list'][fold_idx]

test_score_baseline, test_score_svr, test_score_lasso, test_score_classo, test_score_rf = compare_inner_fold_centralparksoil.one_fold_part1(
    X_df, X, y, label, tr, te, param_grid_lasso, param_grid_kr, param_grid_rf)
test_score_kb, test_score_wkb_ma, test_score_wkb_mb = compare_inner_fold_centralparksoil.one_fold_part2(
    X_df, y, tr, te,
    mod_with_params, weighted_kmat_with_params_ma, weighted_kmat_with_params_mb,
    param_grid_svm, param_grid_rf, param_grid_baseline,
    do_save, do_save_file, do_save_file_weighted_ma, do_save_file_weighted_mb)

scores_series = pd.Series({
    'baseline': test_score_baseline,
    'SVR(RBF)': test_score_svr,
    'Lasso': test_score_lasso,
    'classo': test_score_classo,
    'RF': test_score_rf,
    'KernelBiome': test_score_kb,
    'KB(UFMA)': test_score_wkb_ma,
    'KB(UFMB)': test_score_wkb_mb
})

scores_series.to_csv(
    f"centralparksoil_classo_{pseudo_count}_prescreen_manual_fold_{fold_idx}.csv", index=False)
