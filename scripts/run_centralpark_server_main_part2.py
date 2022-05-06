"""
KernelBiome
"""
# train with the top features
X_tr = X_df.to_numpy().astype('float').T[tr]
train_scores_all, test_scores_all, selected_params_all = run_experiments(X_tr,
                                                                         y[tr],
                                                                         mod_with_params,
                                                                         param_grid_kr,
                                                                         param_grid_rf,
                                                                         param_grid_baseline,
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
                                                                         do_save_filename=do_save_file)
best_models = top_models_in_each_group(
    mod_with_params, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
model_selected = best_models.iloc[0]
print(model_selected)

# refit best model
X_tr /= X_tr.sum(axis=1)[:, None]
pred_fun, gscv = refit_best_model(X_tr, y[tr], 'KernelRidge', param_grid_kr, model_selected,
                                  'neg_mean_squared_error', center_kmat=True, n_fold=5, n_jobs=-1, verbose=0)
print(gscv.best_estimator_)
X_te = X_df.to_numpy().astype('float').T[te]
X_te /= X_te.sum(axis=1)[:, None]
print(X_te.shape)
test_score_kb = np.mean((pred_fun(X_te)-y[te])**2)
print(f"* Done kb: {test_score_kb}.")

"""
Weighted KernelBiome (Original Unifrac weights)
"""
X_tr = X_df.to_numpy().astype('float').T[tr]
X_tr /= X_tr.sum(axis=1)[:, None]
train_scores_all, test_scores_all, selected_params_all = run_experiments(X_tr,
                                                                         y[tr],
                                                                         weighted_mod_with_params_orig,
                                                                         param_grid_kr,
                                                                         param_grid_rf,
                                                                         param_grid_baseline,
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
    weighted_mod_with_params_orig, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
model_selected = best_models.iloc[0]
print(model_selected)
pred_fun, gscv = refit_best_model(X_tr, y[tr],
                                  'KernelRidge',
                                  param_grid_kr,
                                  model_selected,
                                  'neg_mean_squared_error',
                                  center_kmat=True,
                                  n_fold=5,
                                  n_jobs=-1,
                                  verbose=0)
print(gscv.best_estimator_)
X_te = X_df.to_numpy().astype('float').T[te]
X_te /= X_te.sum(axis=1)[:, None]
print(X_te.shape)
test_score_wkb_orig = np.mean((pred_fun(X_te) - y[te]) ** 2)
print(f"* Done wkb (orig): {test_score_wkb_orig}.")

"""
Weighted KernelBiome (PSD Unifrac weights)
"""
X_tr = X_df.to_numpy().astype('float').T[tr]
X_tr /= X_tr.sum(axis=1)[:, None]
train_scores_all, test_scores_all, selected_params_all = run_experiments(X_tr,
                                                                         y[tr],
                                                                         weighted_mod_with_params_psd,
                                                                         param_grid_kr,
                                                                         param_grid_rf,
                                                                         param_grid_baseline,
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
    weighted_mod_with_params_psd, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
model_selected = best_models.iloc[0]
print(model_selected)
pred_fun, gscv = refit_best_model(X_tr, y[tr],
                                  'KernelRidge',
                                  param_grid_kr,
                                  model_selected,
                                  'neg_mean_squared_error',
                                  center_kmat=True,
                                  n_fold=5,
                                  n_jobs=-1,
                                  verbose=0)
print(gscv.best_estimator_)
X_te = X_df.to_numpy().astype('float').T[te]
X_te /= X_te.sum(axis=1)[:, None]
print(X_te.shape)
test_score_wkb_psd = np.mean((pred_fun(X_te) - y[te]) ** 2)
print(f"* Done wkb (psd): {test_score_wkb_psd}.")
