# %%
# load libs
# ^^^^^^

on_computerome = True  # nopep8
import sys  # nopep8
if on_computerome:
    sys.path.insert(0, "./KernelBiome/")
else:
    sys.path.insert(0, "../")  # nopep8

from os.path import join
from datetime import date
import numpy as np
from src.kernels_jax import *
from src.cfi import *
from src.utils import *
import load_centralparksoil
import setup_params

seed_num = 2022
today = date.today()

# %%
# set up paths
# ^^^^^^

# input and output paths
if on_computerome:
    data_path = "KernelBiome/data/CentralParkSoil"
    output_path = "KernelBiome/scripts/output"
else:
    data_path = "/Users/hrt620/Documents/projects/kernelbiome_proj/kernelbiome_clean/data/CentralParkSoil"
    output_path = "/Users/hrt620/Documents/projects/kernelbiome_proj/kernelbiome_clean/scripts/output"

# %%
# load data
# ^^^^^^

X_df, y, label = load_centralparksoil.main(
    data_path=data_path, seed_num=seed_num)

# %%
# set up hyperparams and kernel params
# ^^^^^^

param_grid_kr = dict(alpha=list(np.logspace(-7, 1, 5, base=2)))
print(param_grid_kr)

w_unifrac_ma = np.load(
    join(output_path, "centralparksoil_w_unifrac_MA.npy"), allow_pickle=True)
w_unifrac_mb = np.load(
    join(output_path, "centralparksoil_w_unifrac_MB.npy"), allow_pickle=True)

kmat_with_params, weighted_kmat_with_params_ma, weighted_kmat_with_params_mb = setup_params.main(
    X_df, w_unifrac_ma, w_unifrac_mb)

# %%
# 1.1) Unweighed: fit and select top models in each group
# ^^^^^^

X_all = X_df.to_numpy().astype('float').T
train_scores_all, test_scores_all, selected_params_all = run_experiments(X_all,
                                                                         y,
                                                                         kmat_with_params,
                                                                         param_grid_kr,
                                                                         None,
                                                                         None,
                                                                         center_kmat=True,
                                                                         fac_grid=None,
                                                                         n_fold_outer=10,
                                                                         n_fold_inner=5,
                                                                         type='regression',
                                                                         scoring='neg_mean_squared_error',
                                                                         kernel_estimator='kr',
                                                                         n_jobs=-1,
                                                                         random_state=seed_num,
                                                                         verbose=0)
best_models = top_models_in_each_group(
    kmat_with_params, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
best_models.to_csv(
    join(output_path, f"centralparksoil_unweighted_best_models_{today.strftime('%b-%d-%Y')}.csv"), index=False)

# %%
# 1.2) Unweighted: refit the best kernel
# ^^^^^^

X_all /= X_all.sum(axis=1)[:, None]
model_selected = best_models.iloc[0]
print(model_selected)
pred_fun, gscv = refit_best_model(X_all, y, 'KernelRidge', param_grid_kr, model_selected,
                                  'neg_mean_squared_error', center_kmat=True, n_fold=5, n_jobs=-1, verbose=0)
print(gscv.best_estimator_)

# %%
# 1.3) Unweighted: calculate CFI and CPD values
# ^^^^^^^

df = df_ke_dual_mat(X_all, X_all, gscv.best_estimator_.dual_coef_, range(
    X_all.shape[0]), kernel_args_str_to_k_fun(model_selected.estimator_key))
centralparksoil_cfi_vals = get_perturbation_cfi_index(X_all, df)
centralparksoil_cpd_vals = get_cfi(X_all, X_all, pred_fun)

np.save(join(output_path, "centralparksoil_unweighted_cfi_df.npy"), df)
np.save(join(output_path, "centralparksoil_unweighted_cfi_vals.npy"),
        centralparksoil_cfi_vals)
np.save(join(output_path, "centralparksoil_unweighted_cpd_vals.npy"),
        centralparksoil_cpd_vals)

# %%
# 2.1) Weighted (MA): fit and select top models in each group
# ^^^^^^

do_save = True
do_save_file_weighted = join(
    output_path, f"centralparksoil_tmp_weighted_kb_{today.strftime('%b-%d-%Y')}.pickle")
X_all = X_df.to_numpy().astype('float').T
X_all /= X_all.sum(axis=1)[:, None]
train_scores_all, test_scores_all, selected_params_all = run_experiments(X_all,
                                                                         y,
                                                                         weighted_kmat_with_params_ma,
                                                                         param_grid_kr,
                                                                         None,
                                                                         None,
                                                                         center_kmat=True,
                                                                         n_fold_outer=10,
                                                                         n_fold_inner=5,
                                                                         type='regression',
                                                                         scoring='neg_mean_squared_error',
                                                                         kernel_estimator='kr',
                                                                         n_jobs=-1,
                                                                         random_state=None,
                                                                         verbose=0,
                                                                         do_save=do_save,
                                                                         do_save_filename=do_save_file_weighted)
best_models = top_models_in_each_group(
    weighted_kmat_with_params_ma, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
best_models.to_csv(
    f"output/centralparksoil_weighted_MA_best_models_{today.strftime('%b-%d-%Y')}.csv", index=False)

# %%
# 2.2) Weighted (MA): refit the best kernel
# ^^^^^^

X_all /= X_all.sum(axis=1)[:, None]
# model_selected = best_models.iloc[0]
model_selected = pd.Series({
    'estimator_key': 'aitchison_weighted_c_1e-07',
    'kmat_fun': wrap(kmat_aitchison_weighted, c=1e-07)
})
print(model_selected)
pred_fun, gscv = refit_best_model(X_all, y, 'KernelRidge', param_grid_kr, model_selected,
                                  'neg_mean_squared_error', center_kmat=True, n_fold=5, n_jobs=-1, verbose=0)
print(gscv.best_estimator_)

# %%
# 2.3) Weighted (MA): calculate CFI and CPD values
# ^^^^^^^

df = df_ke_dual_mat(X_all, X_all,
                    gscv.best_estimator_.dual_coef_,
                    range(X_all.shape[0]),
                    kernel_args_str_to_k_fun(model_selected.estimator_key,
                                             weighted=True,
                                             w_mat=w_unifrac_ma))
np.save(join(output_path, "centralparksoil_weighted_MA_cfi_df.npy"), df)

centralparksoil_cfi_vals = get_perturbation_cfi_index(X_all, df)
np.save(join(output_path, "centralparksoil_weighted_MA_cfi_vals.npy"),
        centralparksoil_cfi_vals)

# only calculate the top 10 species according to weighted CFI to save time
centralparksoil_ma_cfi_vals = np.load(
    join(output_path, "centralparksoil_weighted_MA_cfi_vals.npy"))
selected_comp = np.argsort(-np.abs(centralparksoil_ma_cfi_vals))[:10]

supp_grid = get_supp_grid(X_all, n_grid=20)
centralparksoil_cpd_vals = get_cfi(
    X_all, supp_grid, pred_fun, selected_comp, verbose=True)
np.save(join(output_path, "centralparksoil_weighted_MA_cpd_vals.npy"),
        centralparksoil_cpd_vals)
