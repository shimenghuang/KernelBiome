# from cProfile import label
from matplotlib.pyplot import axis
from numpy.linalg import matrix_rank, cond, eigvals
from numpy.core import finfo
from pickle import TRUE
import pandas as pd
import numpy as np
import numpy.linalg as la
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import KernelCenterer
# import matplotlib.pyplot as plt
import pickle
from collections import Counter
import warnings

from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from itertools import product
import timeit
from src.helpers_jax import wrap
from src.kernels_jax import *
from src.weighted_kernels_jax import *


# ---- helpers for using the return value of top_models_in_each_group ----


def get_internel_kernel_name(kernel_name):
    """
    Map kernel names to internel kernel names used in `k_<internel_name>` and `kmat_<internel_name>`.
    """
    kernel_names_dict = {
        'linear': 'linear',
        'rbf': 'rbf',
        'generalized-js': 'hilbert1',
        'hilbertian': 'hilbert2',
        'aitchison': 'aitchison',
        'aitchison-rbf': 'aitchison_rbf',
        'heat-diffusion': 'hd'
    }
    return kernel_names_dict[kernel_name]


def kernel_args_str_to_dict(kernel_args_key):
    """
    Convert a concatenated string of kernel argument e.g. 'aitchison-rbf_c_0.0001_g_0.1' to a dict.
    """
    split_res = kernel_args_key.split('_', maxsplit=1)
    if len(split_res) == 1:
        return {}
    else:
        kernel_args_str = split_res[1]
        args_list = kernel_args_str.split('_')
        # after removing the name of the kernel the number of elements should be even
        assert(len(args_list) % 2 == 0)
        kernel_args_dict = {}
        for key, val in zip(args_list[0::2], args_list[1::2]):
            kernel_args_dict[key] = float(val)
        return kernel_args_dict


def kernel_args_str_to_k_fun(kernel_args_key):
    """
    Convert a concatenated string of kernel argument to the corresponding kernel function `k_<internel_kernel_name>`.

    e.g. 'aitchison-rbf_c_0.0001_g_0.1' to a warpped function that takes two vectors x and y only.
    """
    internel_kernel_name = get_internel_kernel_name(
        kernel_args_key.split('_', maxsplit=1)[0])
    kernel_args_dict = kernel_args_str_to_dict(kernel_args_key)
    return lambda x, y: eval('k_'+internel_kernel_name)(x, y, **kernel_args_dict)


# ---- pipeline utilities ----

def aggregate_level(X_count, comp_lbl, levels,
                    start_levels=["kingdom", "phylum", "class",
                                  "order", "family", "genus", "species"],
                    start_delim=";"):
    """
    Aggregation of count data on a higher level (does also work for relative data, just remember it has to be )
    Parameters
    ----------
    X_count
    comp_lbl
    levels

    Returns
    -------

    """
    df_label = pd.DataFrame([x.split(start_delim) for x in list(comp_lbl)],
                            columns=start_levels)

    df_merge = pd.concat([df_label, pd.DataFrame(X_count.T)], axis=1)
    df_agg = df_merge.groupby(levels).sum()
    df_agg = df_agg.reset_index()

    df_comp_lbl_agg = df_agg[levels]
    comp_lbl_agg = df_comp_lbl_agg.agg(lambda x: ";".join(x.values), axis=1).T
    X_count_agg = df_agg[[
        col for col in df_agg.columns if col not in levels]].values

    return X_count_agg.T, comp_lbl_agg.values


def get_rbf_bandwidth(m):
    """
    m: value of median of squared euc. dist.
    """
    return [np.sqrt(m), 0.5*m, m, m**1.5, m**2, m**2.5, 10*m, 100*m]


def default_kernel_params_grid(g1=None, g2=None):
    if g1 is None:
        grid_rbf = np.logspace(-2, 2, 5)
    else:
        grid_rbf = get_rbf_bandwidth(g1)
    if g2 is None:
        grid_aitrbf = np.logspace(-2, 2, 5)
    else:
        grid_aitrbf = get_rbf_bandwidth(g2)
    kernel_params_dict = {
        'linear': None,
        'rbf': {'g': grid_rbf},
        # Note: hilbert1, only valid when a >= 1, 0.5 <= b <= a
        'generalized-js': {'a': [1, 10, np.inf], 'b': [0.5, 1, 10, np.inf]},
        # Note: hilbert2, only valid when a >= 1 b <= -1 and not both a, b are inf
        'hilbertian': {'a': [1, 10, np.inf], 'b': [-1, -10, -np.inf]},
        'aitchison': {'c': np.logspace(-7, -3, 5)},
        'aitchison-rbf': {'c': np.logspace(-7, -3, 5), 'g': grid_aitrbf},
        'heat-diffusion': {'t': np.linspace(0.9, 1.1, 5)*0.25/np.pi}
    }
    return kernel_params_dict


def default_weighted_kernel_params_grid(w_unifrac, g1=None, g2=None):
    if g1 is None:
        grid_rbf = np.logspace(-2, 2, 5)
    else:
        grid_rbf = get_rbf_bandwidth(g1)
    if g2 is None:
        grid_aitrbf = np.logspace(-2, 2, 5)
    else:
        grid_aitrbf = get_rbf_bandwidth(g2)
    kernel_params_dict = {
        'linear_weighted': {'w': w_unifrac},
        'rbf_weighted': {'g': grid_rbf, "w": w_unifrac},
        # Note: only valid
        'generalized-js_weighted': {'a': [1, 10, np.inf], 'b': [0.5, 1, 10, np.inf], "w": w_unifrac},
        # Note: only valid when a >= 1 b <= -1 and not both a, b are inf
        'hilbertian_weighted': {'a': [1, 10, np.inf], 'b': [-1, -10, -np.inf], "w": w_unifrac},
        'aitchison_weighted': {'c': np.logspace(-7, -3, 5), 'w': w_unifrac},
        'aitchison-rbf_weighted': {'c': np.logspace(-7, -3, 5), 'g': grid_aitrbf, 'w': w_unifrac},
        'heat-diffusion': {'t': np.linspace(0.9, 1.1, 5)*0.25/np.pi}
    }
    return kernel_params_dict


def get_weighted_kmat_with_params(kernel_params_dict, w_unifrac=None):
    kmat_with_params = {}
    for kname, params in kernel_params_dict.items():
        if kname == 'linear':
            kmat_with_params[kname] = kmat_linear
        if kname == 'linear_weighted':
            kmat_with_params[f'{kname}'] = wrap(
                kmat_linear_weighted, w=w_unifrac)

        if kname == 'rbf':
            for g in params['g']:
                kmat_with_params[f'{kname}_g_{g}'] = wrap(kmat_rbf, g=g)
        if kname == 'rbf_weighted':
            for g in params['g']:
                kmat_with_params[f'{kname}_g_{g}'] = wrap(
                    kmat_rbf_weighted, g=g, w=w_unifrac)

        if kname == 'generalized-js':
            params_ab = list(product(params['a'], params['b']))
            for ab in params_ab:
                if ab[0] >= 1 and ab[1] >= 0.5 and ab[1] <= ab[0]:
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(
                        kmat_hilbert1, a=ab[0], b=ab[1])
        if kname == 'generalized-js_weighted':
            params_ab = list(product(params['a'], params['b']))
            for ab in params_ab:
                if ab[0] >= 1 and ab[1] >= 0.5 and ab[1] <= ab[0]:
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(kmat_hilbert1_weighted, a=ab[0], b=ab[1],
                                                                            w=w_unifrac)
        if kname == 'hilbertian':
            params_ab = list(product(params['a'], params['b']))
            for ab in params_ab:
                if ab[0] >= 1 and ab[1] <= -1 and (not jnp.isinf(ab[0]) or not jnp.isinf(ab[1])):
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(
                        kmat_hilbert2, a=ab[0], b=ab[1])
        if kname == 'hilbertian_weighted':
            params_ab = list(product(params['a'], params['b']))
            for ab in params_ab:
                if ab[0] >= 1 and ab[1] <= -1 and (not jnp.isinf(ab[0]) or not jnp.isinf(ab[1])):
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(kmat_hilbert2_weighted, a=ab[0], b=ab[1],
                                                                            w=w_unifrac)
        if kname == 'aitchison':
            for c in params['c']:
                kmat_with_params[f'{kname}_c_{c}'] = wrap(
                    kmat_aitchison, c_X=c, c_Y=c)
        if kname == 'aitchison_weighted':
            for c in params['c']:
                kmat_with_params[f'{kname}_c_{c}'] = wrap(
                    kmat_aitchison_weighted, c_X=c, c_Y=c, w=w_unifrac)

        if kname == 'aitchison-rbf':
            params_cg = list(product(params['c'], params['g']))
            for cg in params_cg:
                kmat_with_params[f'{kname}_c_{cg[0]}_g_{cg[1]}'] = wrap(kmat_aitchison_rbf, g=cg[1], c_X=cg[0],
                                                                        c_Y=cg[0])
        if kname == 'aitchison-rbf_weighted':
            params_cg = list(product(params['c'], params['g']))
            for cg in params_cg:
                kmat_with_params[f'{kname}_c_{cg[0]}_g_{cg[1]}'] = wrap(kmat_aitchison_rbf_weighted, g=cg[1], c_X=cg[0],
                                                                        c_Y=cg[0], w=w_unifrac)
        if kname == 'heat-diffusion':
            for t in params['t']:
                kmat_with_params[f'{kname}_t_{t}'] = wrap(kmat_hd, t=t)
        if kname == 'heat-diffusion_weighted':
            for t in params['t']:
                kmat_with_params[f'{kname}_t_{t}'] = wrap(
                    kmat_hd_weighted, t=t, w=w_unifrac)

    return kmat_with_params


def get_kmat_with_params(kernel_params_dict):
    kmat_with_params = {}
    for kname, params in kernel_params_dict.items():
        if kname == 'linear':
            kmat_with_params[kname] = kmat_linear
        if kname == 'rbf':
            for g in params['g']:
                kmat_with_params[f'{kname}_g_{g}'] = wrap(kmat_rbf, g=g)
        if kname == 'generalized-js':
            params_ab = list(product(params['a'], params['b']))
            for ab in params_ab:
                if ab[0] >= 1 and ab[1] >= 0.5 and ab[1] <= ab[0]:
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(
                        kmat_hilbert1, a=ab[0], b=ab[1])
        if kname == 'hilbertian':
            params_ab = list(product(params['a'], params['b']))
            for ab in params_ab:
                if ab[0] >= 1 and ab[1] <= -1 and (not jnp.isinf(ab[0]) or not jnp.isinf(ab[1])):
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(
                        kmat_hilbert2, a=ab[0], b=ab[1])
        if kname == 'aitchison':
            for c in params['c']:
                kmat_with_params[f'{kname}_c_{c}'] = wrap(
                    kmat_aitchison, c_X=c, c_Y=c)
        if kname == 'aitchison-rbf':
            params_cg = list(product(params['c'], params['g']))
            for cg in params_cg:
                kmat_with_params[f'{kname}_c_{cg[0]}_g_{cg[1]}'] = wrap(
                    kmat_aitchison_rbf, c_X=cg[0], c_Y=cg[0], g=cg[1])
        if kname == 'heat-diffusion':
            for t in params['t']:
                kmat_with_params[f'{kname}_t_{t}'] = wrap(kmat_hd, t=t)
    return kmat_with_params


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


# ---- compositonal data ----


def center_mat(X):
    """
    Center a square matrix X so that it has both row and column means to be 0.
    """
    n = X.shape[0]
    O = np.ones((n, n))
    I = np.eye(n)
    H = I - O/n
    return np.matmul(np.matmul(H, X), H)


def C_fun(X):
    """
    Normalize a vector or an n x p matrix so that each row sums to 1.
    X: nd.array of shape (n,p) or (p,)
    """
    if X.ndim == 1:
        return X/X.sum()
    else:
        return X/X.sum(axis=1)[:, np.newaxis]


def C_partial(X, j):
    """
    Normalize a vector or an n x p compositional matrix X keeping j-th component(s) unchanged.

    This operation is part of the do-operator on compositional data, which is to 
    renormalize after composition(s) j being modified.

    X: nd.array of shape (n,p) or (p,)
    j: int or a list
    """
    if np.any(X > 1) or np.any(X < 0):
        raise ValueError('X must only contain values between 0 and 1.')
    X_new = X.copy()
    if X_new.ndim == 1:
        p = len(X)
        renorm_idx = np.setdiff1d(range(p), j)
        renorm_fac = (1-X_new[j].sum()) / X_new[renorm_idx].sum()
        X_new[renorm_idx] *= renorm_fac
    else:
        p = X_new.shape[1]
        renorm_idx = np.setdiff1d(range(p), j)
        if isinstance(j, list):
            renorm_fac = (1-X_new[:, j].sum(axis=1)) / \
                X_new[:, renorm_idx].sum(axis=1)
        else:
            renorm_fac = (1-X_new[:, j]) / \
                X_new[:, renorm_idx].sum(axis=1)
        X_new[:, renorm_idx] *= renorm_fac[:, None]
    return X_new


def run_nested_cv(X_given, y, estimator, estimator_param_grid, scoring, kmat_fun=None,  center_kmat=False, use_count=False, eps=0, n_fold_outer=10, n_fold_inner=5, stratify=True, shuffle=False, random_state=None, n_jobs=6, verbose=0):
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
                if center_mat:
                    K_test = transformer.transform(K_test)
                test_scores[kk] = model.score(K_test, y_test)
            selected_params[kk] = vars(model.best_estimator_)[
                list(estimator_param_grid.keys())[0]]
        except Exception:
            warnings.warn(
                f'Training or scoring failed for fold {kk}. Skipped.')
            pass
    return train_scores, test_scores, selected_params


def make_result_table(mod_with_params, train_scores_all, test_scores_all, selected_params_all):
    kmat_and_params = mod_with_params.keys()
    kmat_and_params = [ss.split('_', maxsplit=1) for ss in kmat_and_params]
    kmat_and_params = pd.DataFrame(kmat_and_params, columns=[
                                   'kernel', 'kernel_params'])
    # index = pd.MultiIndex.from_frame(kmat_and_params)
    res = pd.DataFrame({'avg_train_score': train_scores_all.mean(axis=1),
                        'avg_test_score': test_scores_all.mean(axis=1),
                        'most_freq_best_param': [Counter(row).most_common(1)[0][0] for row in selected_params_all],
                        # 'best_test_score': test_scores_all.max(axis=1),
                        # 'best_test_param': [row[ii] for row, ii in zip(selected_params_all, np.argmax(test_scores_all, axis=1))]
                        })
    res = pd.concat([kmat_and_params, res], axis=1)
    res.sort_values(['kernel', 'avg_test_score'],
                    ascending=[True, False], inplace=True)
    res.set_index(['kernel', 'kernel_params'], inplace=True)
    return res


def top_models_in_each_group(mod_with_params, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=False):
    estimator_keys = list(mod_with_params.keys())
    kmat_and_params = [ss.split('_', maxsplit=1) for ss in estimator_keys]
    kmat_and_params = pd.DataFrame(kmat_and_params, columns=[
                                   'kernel', 'kernel_params'])
    res = pd.DataFrame({'estimator_key': estimator_keys,
                        'kmat_fun': mod_with_params.values(),
                        'avg_train_score': train_scores_all.mean(axis=1),
                        'avg_test_score': test_scores_all.mean(axis=1),
                        'most_freq_best_param': [Counter(row).most_common(1)[0][0] for row in selected_params_all]
                        })
    res = pd.concat([kmat_and_params, res], axis=1)
    # drop RF, baseline, linear and rbf if required
    if kernel_mod_only:
        rf_idx = [ii for ii in range(len(
            estimator_keys)) if 'RF' in estimator_keys[ii] or 'baseline' in estimator_keys[ii]]
        res.drop(rf_idx, inplace=True)
    # get top_n models in each group, models[0].kernel will give the kernel
    res.sort_values(['kernel', 'avg_test_score', 'avg_train_score'], ascending=[
                    True, False, False], inplace=True)
    top_res = res.groupby('kernel')[
        ['estimator_key', 'kmat_fun', 'avg_test_score', 'most_freq_best_param']].head(top_n)
    return top_res.sort_values('avg_test_score', ascending=False)


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


def Phi(X_new, X_old, kernel_with_params, center=False, pc=0, return_mean=False):
    """
    Projection on the first `pc` number of PCs. 
    """
    # n = X_old.shape[0]
    K_old = kernel_with_params(X_old, X_old)
    transformer = KernelCenterer().fit(K_old)
    K_old_tilde = transformer.transform(K_old) if center else K_old
    # K_old_tilde = center_mat(K_old) if center else K_old
    w_old, V_old = la.eig(K_old_tilde)
    w_old = np.real(w_old)
    V_old = np.real(V_old)
    idx = w_old.argsort()[::-1]
    w_old = w_old[idx]
    V_old = V_old[:, idx]
    K_new = kernel_with_params(X_new, X_old)
    K_new_tilde = transformer.transform(K_new) if center else K_new
    # Q = Q - 1/n*np.ones((n, n)).dot(K_old) - 1/n*Q.dot(np.ones((n, n))) + \
    #     1/n**2*np.ones((n, n)).dot(K_old).dot(np.ones((n, n)))
    if return_mean:
        return K_new_tilde.dot(V_old[:, pc]).mean(axis=0)/np.sqrt(w_old[pc])
    else:
        return K_new_tilde.dot(V_old[:, pc])/np.sqrt(w_old[pc])


# ---- depreciated ----

# def _cfii_plot(cfii_vals, n_digit, axs=None):
#     n_comp = cfii_vals.shape[0]
#     mark_width = int(np.max(cfii_vals)) * 0.01
#     axs.set_ylim(-0.5, n_comp-0.5)
#     axs.vlines(0, ymin=-1, ymax=n_comp, color='gray', linewidth=2, alpha=0.2)
#     for ii in range(n_comp):
#         val = cfii_vals[ii]
#         col = '#ff7f0e' if val > 0 else '#1f77b4'
#         if val != 0:
#             axs.arrow(0, ii, val, 0, width=0.01,
#                       head_width=0.1, color=col, alpha=0.5)
#         axs.annotate(np.round(val, n_digit), xy=(val/2.0, ii+0.1))
#         axs.hlines(ii, xmin=-mark_width, xmax=mark_width,
#                    linewidth=2, color='black')
#         axs.annotate(f'$x^{ii}$', xy=(0+0.05, ii-0.2))

#     axs.spines['top'].set_visible(False)
#     axs.spines['right'].set_visible(False)
#     axs.spines['bottom'].set_visible(False)
#     axs.spines['left'].set_visible(False)
#     # axs.get_xaxis().set_visible(False)
#     axs.axis('off')
#     return axs
