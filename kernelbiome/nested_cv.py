import timeit
import warnings
import pickle
import numpy as np
from numpy.linalg import matrix_rank, eigvals
from numpy.core import finfo
from sklearn.preprocessing import KernelCenterer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier


# ---- helper functions ----


def get_kr_penalty(X, kmat_fun, alpha_max=5.0, n_grid=5, name=None, verbose=False):
    """
    Make sure the penalty alpha is greater than the rank detection threshold.
    Adding identity matrix times these to K should ensure K to be pd.
    See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html
    """
    K = kmat_fun(X, X)
    if np.isnan(K).any() or np.isinf(K).any() or (K == 0.0).all():
        print("K contains NaN or Inf, or it is all 0s (heat-diffusion numerical issue).")
        return {'alpha': []}
    else:
        K_eigs = np.real(eigvals(K))
        rank_tol = K_eigs.max() * K.shape[0] * finfo(K.dtype).eps
        scale_tol = np.floor(np.log10(rank_tol))  # e.g. 2e-5 would be -5
        # Note: we might to add something strictly greater than rank_tol?
        alpha_dic = {'alpha': np.linspace(
            rank_tol, max(5*rank_tol, alpha_max), n_grid)}
        if verbose:
            print(f"rank_tol: {rank_tol}")
            print(f"scale_tol: 10^{scale_tol}")
            print(f"{name} {alpha_dic}")
        return alpha_dic


def check_kr_penalty(X, kmat_fun, param_grid, name=None, verbose=True, verbose_rank=False):
    """
    Drop alpha values that does not guarantee K + alpha*I invertible.
    """
    alpha_keep = []
    for alpha in param_grid['alpha']:
        K = kmat_fun(X, X) + alpha*np.eye(X.shape[0])
        try:
            rK = matrix_rank(K, hermitian=True)
            if verbose_rank:
                print(f"alpha = {alpha}: rank K is {rK}.")
            if rK == X.shape[0]:
                alpha_keep.append(alpha)
        except Exception:
            print(f"{name} rK not calculated properly, this alpha is dropped.")
            pass

    if len(alpha_keep) > 0:
        alpha_dict = {'alpha': alpha_keep}
        if verbose:
            print(f"Checked alpha grid: {alpha_dict}")
        return alpha_dict
    else:
        print("None of the alpha values worked! Returning None.")
        return None


# ---- main functions ----


def run_nested_cv(X_given, y, estimator, estimator_param_grid, scoring, kmat_fun=None, center_kmat=False, use_count=False, eps=0,
                  n_fold_outer=10, n_fold_inner=5, stratify=True, shuffle=False, random_state=None, n_jobs=6, verbose=0):
    # if not shuffle random_state should not be used.
    random_state = None if not shuffle else random_state
    kf = StratifiedKFold(n_splits=n_fold_outer, shuffle=shuffle, random_state=random_state) if stratify else KFold(
        n_splits=n_fold_outer, shuffle=shuffle)
    train_scores = np.full(n_fold_outer, np.NaN)
    test_scores = np.full(n_fold_outer, np.NaN)
    selected_params = np.full(n_fold_outer, np.NaN, dtype=object)
    if use_count:
        X = X_given.copy() + eps
    else:
        X = X_given.copy() + eps
        X /= X.sum(axis=1)[:, None]
    for kk, (train_index, test_index) in enumerate(kf.split(X, y)):
        if verbose and kk % 1 == 0:
            print(f"fold {kk+1} out of {n_fold_outer} folds")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        try:
            model = GridSearchCV(estimator=estimator, param_grid=estimator_param_grid, cv=n_fold_inner,
                                 scoring=scoring, n_jobs=n_jobs, verbose=verbose)
            if kmat_fun is None:
                model.fit(X_train, y_train)
                train_scores[kk] = model.score(X_train, y_train)
                test_scores[kk] = model.score(X_test, y_test)
            else:
                K_train = kmat_fun(X_train, X_train)
                transformer = KernelCenterer().fit(K_train)
                if center_kmat:
                    K_train = transformer.transform(K_train)
                model.fit(K_train, y_train)
                train_scores[kk] = model.score(K_train, y_train)
                K_test = kmat_fun(X_test, X_train)
                if center_kmat:
                    K_test = transformer.transform(K_test)
                test_scores[kk] = model.score(K_test, y_test)
            selected_params[kk] = vars(model.best_estimator_)[
                list(estimator_param_grid.keys())[0]]
        except Exception:
            warnings.warn(
                f'Training or scoring failed for fold {kk}. Skipped.')
            pass
    return train_scores, test_scores, selected_params


def run_experiments(X, y, mod_with_params, param_grid_ke, param_grid_rf=None, param_grid_bl=None, center_kmat=False, fac_grid=None,
                    n_fold_outer=10, n_fold_inner=5, type='regression', scoring='neg_mean_squared_error', kernel_estimator='kr',
                    n_jobs=6, random_state=None, verbose=0, do_save=False, do_save_filename="res.pickle", print_each=True):
    """
    Parameters:
    mod_with_params: dict
        A dict containing kernel names as keys and kernel matrix functions as values, such as the object returned by `get_kmat_with_params`. One can also add other models such as
        >>> # kernel models
        >>> kernel_params_dict = default_kernel_params_grid()
        >>> kmat_with_params = get_kmat_with_params(kernel_params_dict)
        >>> # add RF and baseline
        >>> mod_with_params = kmat_with_params
        >>> mod_with_params['RF use comp'] = None
        >>> mod_with_params['baseline'] = None
    param_grid_ke: dict
        A dict containing values for inner CV hyperparameter selection of a kernel estimator (KernelRidge, svm.SVR or svm.SVC).
    param_grid_rf: dict
        A dict containing values for inner CV hyperparameter selection of a random forest estimator (RandomForestRegressor or RandomForestClassifier).
    param_grid_bl: dict
        A dict containing values for inner CV hyperparameter selection of a baseline estimator (DummyRegressor or DummyClassifier).
    center_kmat: bool
        Whether to center the kernel matrix before fitting.
    """
    n_estimator = len(mod_with_params)
    train_scores_all = np.full((n_estimator, n_fold_outer), np.nan)
    test_scores_all = np.full((n_estimator, n_fold_outer), np.nan)
    selected_params_all = np.full(
        (n_estimator, n_fold_outer), np.nan, dtype=object)
    time_estimators = np.full(n_estimator, np.nan)

    if type == 'regression':
        if kernel_estimator not in ['kr', 'svm']:
            raise ValueError("`kernel_estimator` can only be 'kr' or 'svm'.")
        RF = RandomForestRegressor
        Dummy = DummyRegressor
        KE = KernelRidge if kernel_estimator == 'kr' else svm.SVR
        stratify = False
    elif type == 'classification':
        RF = RandomForestClassifier
        Dummy = DummyClassifier
        KE = svm.SVC
        stratify = True
    else:
        raise ValueError(
            "`type` can only be 'regression' or 'classification'.")

    for ii, (name, fun) in enumerate(mod_with_params.items()):
        if print_each:
            print(f"--- running: {name} ---")

        if name == 'RF use comp':
            estimator = RF(max_depth=np.sqrt(X.shape[0]))
            time_estimators[ii] = timeit.default_timer()
            train_scores_all[ii, :], test_scores_all[ii, :], selected_params_all[ii, :] = run_nested_cv(
                X, y, estimator, param_grid_rf, scoring, kmat_fun=None, center_kmat=center_kmat, use_count=False, eps=0, n_fold_outer=n_fold_outer, n_fold_inner=n_fold_inner, stratify=stratify, shuffle=False, random_state=random_state, n_jobs=n_jobs, verbose=verbose)
            time_estimators[ii] = timeit.default_timer()-time_estimators[ii]
        elif name == 'RF use count':
            estimator = RF(max_depth=np.sqrt(X.shape[0]))
            time_estimators[ii] = timeit.default_timer()
            train_scores_all[ii, :], test_scores_all[ii, :], selected_params_all[ii, :] = run_nested_cv(
                X, y, estimator, param_grid_rf, scoring, kmat_fun=None, center_kmat=center_kmat, use_count=True, eps=0, n_fold_outer=n_fold_outer, n_fold_inner=n_fold_inner, stratify=stratify, shuffle=False, random_state=random_state, n_jobs=n_jobs, verbose=verbose)
            time_estimators[ii] = timeit.default_timer()-time_estimators[ii]
        elif name == 'baseline':
            estimator = Dummy()
            time_estimators[ii] = timeit.default_timer()
            train_scores_all[ii, :], test_scores_all[ii, :], selected_params_all[ii, :] = run_nested_cv(
                X, y, estimator, param_grid_bl, scoring, kmat_fun=None, center_kmat=center_kmat, use_count=False, eps=0, n_fold_outer=n_fold_outer, n_fold_inner=n_fold_inner, stratify=stratify, shuffle=False, random_state=random_state, n_jobs=n_jobs, verbose=verbose)
            time_estimators[ii] = timeit.default_timer()-time_estimators[ii]
        else:
            estimator = KE(kernel='precomputed')
            if kernel_estimator == 'kr':
                intercept = np.mean(y)
                y = y - intercept
                if not center_kmat:
                    warnings.warn(
                        'Using KernelRidge without centeralizing kernel matrix might affect performance.')
                # New: drop bad alphas
                X_comp = X.copy()
                X_comp /= X_comp.sum(axis=1)[:, None]
                param_grid_ke_use = get_kr_penalty(
                    X_comp, fun, alpha_max=5.0, n_grid=5, name=name, verbose=True)
                param_grid_ke_use = check_kr_penalty(
                    X_comp, fun, param_grid_ke_use, name, verbose=True)
            else:
                param_grid_ke_use = param_grid_ke
            time_estimators[ii] = timeit.default_timer()
            if param_grid_ke_use is not None:
                train_scores_all[ii, :], test_scores_all[ii, :], selected_params_all[ii, :] = run_nested_cv(
                    X, y, estimator, param_grid_ke_use, scoring, kmat_fun=fun, center_kmat=center_kmat, use_count=False, eps=0, n_fold_outer=n_fold_outer, n_fold_inner=n_fold_inner, stratify=stratify, shuffle=False, random_state=random_state, n_jobs=n_jobs, verbose=verbose)
            else:
                print(f"`param_grid_ke_use` is None. Run skipped.")
            time_estimators[ii] = timeit.default_timer()-time_estimators[ii]

        if print_each:
            print(f"average test score: {np.mean(test_scores_all[ii, :])}")

        if do_save:
            res = {}
            res.update({"train_scores_all": train_scores_all,
                        "test_scores_all": test_scores_all,
                        "selected_params_all": selected_params_all})

            with open(do_save_filename, "wb") as handle:
                pickle.dump(res, handle)
            handle.close()

        if verbose:
            print(
                "---- Done {0}. Time: {1:.3f} sec.  ----".format(name, time_estimators[ii]))

    return train_scores_all, test_scores_all, selected_params_all


def fit_model(X, y, estimator_name, estimator_param_grid, kmat_fun, scoring, center_kmat=True, fac_grid=None, n_fold=5, n_jobs=6, verbose=0):
    """
    estimator_name: str
        one of 'KernelRidge', 'SVR', or 'SVC'.
    kmat_fun:
        wrapped kernel matrix function which only takes X, Y.
    """
    if estimator_name == "KernelRidge":
        estimator = KernelRidge(kernel="precomputed")
        estimator_param_grid_use = get_kr_penalty(
            X, kmat_fun, alpha_max=5.0, n_grid=5, name=None, verbose=True)
        estimator_param_grid_use = check_kr_penalty(
            X, kmat_fun, estimator_param_grid_use, verbose=False)
    elif estimator_name == "SVR":
        estimator = svm.SVR(kernel="precomputed")
        estimator_param_grid_use = estimator_param_grid
    elif estimator_name == "SVC":
        estimator = svm.SVC(kernel="precomputed", probability=True)
        estimator_param_grid_use = estimator_param_grid
    else:
        raise ValueError(
            "estimator_name can only be one of 'KernelRidge', 'SVR', or 'SVC'.")
    gscv = GridSearchCV(estimator=estimator, param_grid=estimator_param_grid_use, cv=n_fold,
                        scoring=scoring, n_jobs=n_jobs, verbose=verbose)
    K = kmat_fun(X, X)
    transformer = KernelCenterer().fit(K)
    if center_kmat:
        K = transformer.transform(K)
    if estimator_name == "KernelRidge":
        intercept = np.mean(y)
        gscv.fit(K, y-intercept)

        def pred_fun(X_new, center_kmat=center_kmat):
            K = kmat_fun(X_new, X)
            if center_kmat:
                K = transformer.transform(K)
            return gscv.best_estimator_.predict(K) + intercept
    elif estimator_name == "SVR":
        gscv.fit(K, y)

        def pred_fun(X_new):
            K = kmat_fun(X_new, X)
            return gscv.best_estimator_.predict(K)
    elif estimator_name == "SVC":
        gscv.fit(K, y)

        def pred_fun(X_new):
            K = kmat_fun(X_new, X)
            # Note: for classification, returning the probabilities of having value 1
            return gscv.best_estimator_.predict_proba(K)[:, 1]
    return pred_fun, gscv


def refit_best_model(X, y, estimator_name, estimator_param_grid, best_model, scoring, center_kmat=True, n_fold=5, n_jobs=6, verbose=0):
    return fit_model(X, y, estimator_name, estimator_param_grid, best_model.kmat_fun, scoring, center_kmat=center_kmat, n_fold=n_fold, n_jobs=n_jobs, verbose=verbose)
