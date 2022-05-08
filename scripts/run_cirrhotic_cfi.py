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
from src.weighted_kernels_jax import *
from src.cfi import *
from src.utils import *
import load_cirrhotic
import setup_params

seed_num = 2022
today = date.today()

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

# %%
# load data
# ^^^^^^

X_df, y, label = load_cirrhotic.main(data_path=data_path, seed_num=seed_num)

# %%
# set up hyperparams and kernel params
# ^^^^^^

param_grid_svm = dict(C=[10**x for x in [-3, -2, -1, 0, 1, 2, 3]])
print(param_grid_svm)

w_unifrac_ma = np.load(
    join(output_path, "cirrhotic_w_unifrac_MA.npy"), allow_pickle=True)
w_unifrac_mb = np.load(
    join(output_path, "cirrhotic_w_unifrac_MB.npy"), allow_pickle=True)

kmat_with_params, weighted_kmat_with_params_ma, weighted_kmat_with_params_mb = setup_params.main(
    X_df, w_unifrac_ma, w_unifrac_mb)

# %%
# 1.1) Unweighed: fit and select top models in each group
# ^^^^^^

X_all = X_df.to_numpy().astype('float').T
train_scores_all, test_scores_all, selected_params_all = run_experiments(X_all,
                                                                         y,
                                                                         kmat_with_params,
                                                                         param_grid_svm,
                                                                         None,
                                                                         None,
                                                                         center_kmat=False,
                                                                         fac_grid=None,
                                                                         n_fold_outer=10,
                                                                         n_fold_inner=5,
                                                                         type='classification',
                                                                         scoring='accuracy',
                                                                         kernel_estimator='svm',
                                                                         n_jobs=-1,
                                                                         random_state=seed_num,
                                                                         verbose=0)
best_models = top_models_in_each_group(
    kmat_with_params, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
best_models.to_csv(
    join(output_path, f"cirrhotic_unweighted_best_models_{today.strftime('%b-%d-%Y')}.csv"), index=False)


# %%
# 1.2) Unweighted: refit the best kernel
# ^^^^^^

X_all /= X_all.sum(axis=1)[:, None]
model_selected = best_models.iloc[0]
print(model_selected)
pred_fun, gscv = refit_best_model(X_all, y, 'SVC', param_grid_svm, model_selected,
                                  'accuracy', center_kmat=False, n_fold=5, n_jobs=-1, verbose=0)
print(gscv.best_estimator_)

# %%
# 1.3) Unweighted: calculate CFI and CPD values
# ^^^^^^^

df = df_ke_dual_mat(X_all, X_all, gscv.best_estimator_.dual_coef_[
                    0], gscv.best_estimator_.support_,
                    kernel_args_str_to_k_fun(model_selected.estimator_key))
cirrhotic_cfi_vals = get_perturbation_cfi_index(X_all, df)
cirrhotic_cpd_vals = get_cfi(X_all, X_all, pred_fun)

np.save(join(output_path, "cirrhotic_unweighted_cfi_df.npy"), df)
np.save(join(output_path, "cirrhotic_unweighted_cfi_vals.npy"), cirrhotic_cfi_vals)
np.save(join(output_path, "cirrhotic_unweighted_cpd_vals.npy"), cirrhotic_cpd_vals)


# %%
# 2.1) Weighted (MA): fit and select top models in each group
# ^^^^^^

do_save = True
do_save_file_weighted = join(
    output_path, f"cirrhotic_tmp_weighted_kb_{today.strftime('%b-%d-%Y')}.pickle")
X_all = X_df.to_numpy().astype('float').T
X_all /= X_all.sum(axis=1)[:, None]
train_scores_all, test_scores_all, selected_params_all = run_experiments(X_all,
                                                                         y,
                                                                         weighted_kmat_with_params_ma,
                                                                         param_grid_svm,
                                                                         None,
                                                                         None,
                                                                         center_kmat=False,
                                                                         n_fold_outer=10,
                                                                         n_fold_inner=5,
                                                                         type='classification',
                                                                         scoring='accuracy',
                                                                         kernel_estimator='SVC',
                                                                         n_jobs=-1,
                                                                         random_state=seed_num,
                                                                         verbose=0,
                                                                         do_save=do_save,
                                                                         do_save_filename=do_save_file_weighted)
best_models = top_models_in_each_group(
    weighted_kmat_with_params_ma, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
best_models.to_csv(
    f"cirrhotic_weighted_MA_best_models_{today.strftime('%b-%d-%Y')}.csv", index=False)

# %%
# 2.2) Weighted (MA): refit the best kernel
# ^^^^^^

X_all /= X_all.sum(axis=1)[:, None]
model_selected = best_models.iloc[0]
print(model_selected)
pred_fun, gscv = refit_best_model(X_all, y, 'SVC', param_grid_svm, model_selected,
                                  'accuracy', center_kmat=False, n_fold=5, n_jobs=-1, verbose=0)
print(gscv.best_estimator_)

# %%
# 2.3) Weighted (MA): calculate CFI and CPD values
# ^^^^^^^

df = df_ke_dual_mat(X_all, X_all,
                    gscv.best_estimator_.dual_coef_[0],
                    gscv.best_estimator_.support_,
                    kernel_args_str_to_k_fun(model_selected.estimator_key,
                                             weighted=True,
                                             w_mat=w_unifrac_ma))
np.save(join(output_path, "cirrhotic_weighted_MA_cfi_df.npy"), df)

cirrhotic_cfi_vals = get_perturbation_cfi_index(X_all, df)
np.save(join(output_path, "cirrhotic_weighted_MA_cfi_vals.npy"), cirrhotic_cfi_vals)

cirrhotic_cpd_vals = get_cfi(X_all, X_all, pred_fun)
np.save(join(output_path, "cirrhotic_weighted_MA_cpd_vals.npy"), cirrhotic_cpd_vals)

# %%
