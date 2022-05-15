import sys  # nopep8
sys.path.insert(0, "../")  # nopep8

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from classo import classo_problem
from src.utils import *


def one_fold_part1(X_df, X, y, label, tr, te, param_grid_baseline, param_grid_lasso, param_grid_svr_rbf, param_grid_rf):

    X_tr = X_df.to_numpy().astype('float').T[tr]
    X_tr /= X_tr.sum(axis=1)[:, None]

    X_te = X_df.to_numpy().astype('float').T[te]
    X_te /= X_te.sum(axis=1)[:, None]

    """
    baseline
    """
    estimator = DummyRegressor()
    gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_baseline, cv=5,
                        scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
    gscv.fit(X_tr, y[tr])
    yhat = gscv.predict(X_te)
    test_score_baseline = np.mean((yhat-y[te])**2)
    print(f"* Done baseline: {test_score_baseline}.")

    """
    SVR with RBF
    """
    estimator = SVR()
    gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_svr_rbf, cv=5,
                        scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
    gscv.fit(X_tr, y[tr])
    print(gscv.best_estimator_)
    yhat = gscv.predict(X_te)
    test_score_svr = np.mean((yhat-y[te])**2)
    print(f"* Done SVC with RBF: {test_score_svr}.")

    """
    Lasso
    """
    estimator = Lasso(random_state=2022)
    gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_lasso, cv=5,
                        scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
    gscv.fit(X_tr, y[tr])
    print(gscv.best_estimator_)
    yhat = gscv.predict(X_te)
    test_score_lasso = np.mean((yhat-y[te])**2)
    print(f"* Done Lasso: {test_score_lasso}.")

    """
    c-lasso
    """
    problem = classo_problem(X[tr], y[tr], label=label)
    # problem.formulation.w = 1 / nleaves
    problem.formulation.intercept = True
    problem.formulation.concomitant = False
    problem.model_selection.StabSel = False
    problem.model_selection.PATH = True
    problem.model_selection.CV = True
    # one could change logscale, Nsubset, oneSE
    problem.model_selection.CVparameters.seed = (6)
    problem.solve()
    alpha = problem.solution.CV.refit
    yhat = X[te].dot(alpha[1:]) + alpha[0]
    test_score_classo = np.mean((yhat-y[te])**2)
    print(f"* Done classo: {test_score_classo}.")

    """
    RF
    """
    estimator = RandomForestRegressor(max_depth=np.sqrt(X.shape[0]))
    gscv = GridSearchCV(estimator=estimator, param_grid=param_grid_rf, cv=5,
                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    gscv.fit(X_tr, y[tr])
    test_score_rf = np.mean((gscv.predict(X_te)-y[te])**2)
    print(f"* Done rf: {test_score_rf}.")

    return test_score_baseline, test_score_svr, test_score_lasso, test_score_classo, test_score_rf


def one_fold_part2(X_df, y, tr, te,
                   mod_with_params, weighted_kmat_with_params_ma, weighted_kmat_with_params_mb,
                   param_grid_kr, param_grid_rf, param_grid_baseline,
                   do_save, do_save_file, do_save_file_weighted_ma, do_save_file_weighted_mb):
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
                                                                             weighted_kmat_with_params_ma,
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
                                                                             do_save_filename=do_save_file_weighted_ma)
    best_models = top_models_in_each_group(
        weighted_kmat_with_params_ma, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
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
    test_score_wkb_ma = np.mean((pred_fun(X_te) - y[te]) ** 2)
    print(f"* Done wkb (orig): {test_score_wkb_ma}.")

    """
    Weighted KernelBiome (PSD Unifrac weights)
    """
    X_tr = X_df.to_numpy().astype('float').T[tr]
    X_tr /= X_tr.sum(axis=1)[:, None]
    train_scores_all, test_scores_all, selected_params_all = run_experiments(X_tr,
                                                                             y[tr],
                                                                             weighted_kmat_with_params_mb,
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
                                                                             do_save_filename=do_save_file_weighted_mb)
    best_models = top_models_in_each_group(
        weighted_kmat_with_params_mb, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
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
    test_score_wkb_mb = np.mean((pred_fun(X_te) - y[te]) ** 2)
    print(f"* Done wkb (psd): {test_score_wkb_mb}.")

    return test_score_kb, test_score_wkb_ma, test_score_wkb_mb
