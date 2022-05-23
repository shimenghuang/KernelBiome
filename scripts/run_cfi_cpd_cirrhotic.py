# %%
# load libs
# ^^^^^^

from pathlib import Path  # nopep8
import sys  # nopep8
path_root = Path(__file__).parents[1]  # nopep8
sys.path.append(str(path_root))  # nopep8

from os.path import join
from datetime import date
import numpy as np
from kernelbiome.kernels_jax import *
from kernelbiome.kernels_weighted_jax import *
from kernelbiome.nested_cv import *
from kernelbiome.cfi_and_cpd import *
from kernelbiome.utils_cv import *
from kernelbiome.utils_result import *
import load_cirrhotic
import setup_params

seed_num = 2022
today = date.today()

# %%
# set up paths
# ^^^^^^

on_computerome = True  # nopep8

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
    join(output_path, f"cirrhotic_weighted_MA_best_models_{today.strftime('%b-%d-%Y')}.csv", index=False))

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
# 3.1) Artifically weighted CFI
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

X_keep = X_df.to_numpy().astype('float').T
X_keep = X_keep/X_keep.sum(axis=1)[:, None]  # 102, 36
dist_mat = np.ones(
    (X_keep.shape[1], X_keep.shape[1])) - np.eye(X_keep.shape[1])
dist_mat[[102, 36], [36, 102]] = 0

M = 1.0-dist_mat
D = np.diag(1.0/np.sqrt(M.sum(axis=1)))
W1 = D.dot(M).dot(D)

param_grid_svm = dict(C=[10**x for x in [-3, -2, -1, 0, 1, 2, 3]])
print(param_grid_svm)

weighted_kernel_params_dict_art = default_weighted_kernel_params_grid(
    W1, g1, g2)
weighted_kmat_with_params_art = get_weighted_kmat_with_params(
    weighted_kernel_params_dict_art, w_unifrac=W1)

train_scores_all, test_scores_all, selected_params_all = run_experiments(X_keep,
                                                                         y,
                                                                         weighted_kmat_with_params_art,
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
                                                                         random_state=None,
                                                                         verbose=0,
                                                                         do_save=False)
best_models = top_models_in_each_group(
    weighted_kmat_with_params_art, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
best_models.to_csv(
    join(output_path, f"cirrhotic_weighted_art_best_models_{today.strftime('%b-%d-%Y')}.csv"), index=False)

# %%
# 3.2) refit best model
# ^^^^^^

model_selected = best_models.iloc[0]
print(model_selected)
# refit model
pred_fun, gscv = refit_best_model(X_keep, y, 'SVC', param_grid_svm, model_selected,
                                  'accuracy', center_kmat=False, n_fold=5, n_jobs=6, verbose=0)
print(gscv.best_estimator_)

df = df_ke_dual_mat(X_keep, X_keep, gscv.best_estimator_.dual_coef_[
    0], gscv.best_estimator_.support_, wrap(k_aitchison_weighted, c=1e-5, w=W1))
cfi_vals_art = get_perturbation_cfi_index(X_keep, df)

# %%
# 3.3) Artificially weighted: calculate CFI and CPD values
# ^^^^^^^

df = df_ke_dual_mat(X_comp, X_comp,
                    gscv.best_estimator_.dual_coef_[0],
                    gscv.best_estimator_.support_,
                    kernel_args_str_to_k_fun(model_selected.estimator_key,
                                             weighted=True,
                                             w_mat=W1))
np.save(join(output_path, "cirrhotic_weighted_artificial_cfi_df.npy"), df)

cirrhotic_cfi_vals = get_perturbation_cfi_index(X_comp, df)
np.save(join(output_path, "cirrhotic_weighted_artificial_cfi_vals.npy"),
        cirrhotic_cfi_vals)

# cirrhotic_cpd_vals = get_cfi(X_all, X_all, pred_fun)
# np.save(join(output_path, "cirrhotic_weighted_artificial_cpd_vals.npy"),
#         cirrhotic_cpd_vals)

# %%
