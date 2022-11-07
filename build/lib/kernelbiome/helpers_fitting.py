import numpy as np
import pandas as pd
import warnings
from collections import Counter
import jax.numpy as jnp
from kernelbiome.helpers_jax import (wrap, gmean, squared_euclidean_distances)
import kernelbiome.kernels_jax as kj
import kernelbiome.kernels_weighted_jax as wkj
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KernelCenterer


# Function to initialize a default set of kernels
def default_kernel_models(X):
    # Compute minimal X value
    minX = X[X != 0].min()
    # Compute parameters for rbf
    K = squared_euclidean_distances(X, X)
    k_triu = K[np.triu_indices(n=K.shape[0], k=1, m=K.shape[1])]
    g1 = 1.0/np.median(k_triu)
    grid_rbf = g1*np.logspace(-4, 2, 7)
    # Compute parameters for generalized-js
    grid_gjs = [[1, 0.5], [1, 1],
                [10, 0.5], [10, 1], [10, 10],
                [np.inf, 0.5], [np.inf, 1], [np.inf, 10], [np.inf, np.inf]]
    # Compute parameters for hilbertian
    grid_hb = [[1, -1], [1, -10], [1, -np.inf],
               [10, -1], [10, -10], [10, -np.inf],
               [np.inf, -1], [np.inf, -10]]
    # Compute parameters for aitchison
    grid_c = np.geomspace(minX/2 * 1e-04,
                          np.min([minX/2 * 1e+04, 1e-02]), 9)
    # Compute parameters for aitchison-rbf
    zero_grid = np.geomspace(minX/2 * 1e-04,
                             np.min([minX/2 * 1e+04, 1e-02]), 5)
    grid_cg = []
    for zz in zero_grid:
        Xc = X + zz
        gm_Xc = gmean(Xc, axis=1)
        clr_Xc = np.log(Xc/gm_Xc[:, None])
        K = squared_euclidean_distances(clr_Xc, clr_Xc)
        k_triu = K[np.triu_indices(n=K.shape[0], k=1, m=K.shape[1])]
        g2 = 1.0/np.median(k_triu)
        grid_aitrbf = g2*np.logspace(-1, 1, 3)
        for gg in grid_aitrbf:
            grid_cg.append([zz, gg])
    # Compute parameters for heat-diffusion
    grid_t = (1/(np.logspace(-20, 1, 6))**(2/(X.shape[1]-1)))*0.25/np.pi
    # Specify models
    models = {
        'linear': None,
        'rbf': {'g': grid_rbf},
        'generalized-js': {'ab': grid_gjs},
        'hilbertian': {'ab': grid_hb},
        'aitchison': {'c': grid_c},
        'aitchison-rbf': {'cg': grid_cg},
        'heat-diffusion': {'t': grid_t}
    }
    return models


# Function to contruct default hyperparameter grid
def get_hyperpar_grid(X, kmat_fun, estimator, n_grid=10, verbose=0):
    estimator_name = estimator.__class__.__name__

    K = kmat_fun(X, X)
    # Return None if error in kernel matrix
    if np.isnan(K).any() or np.isinf(K).any() or (K == 0.0).all():
        warnings.warn("K contains NaN or Inf, or is all 0s" +
                      "(e.g. heat-diffuision numerical issues")
        return None
    # Select grid based on spectrum of kernel
    K_eigs = np.real(np.linalg.eigvals(K))
    rank_tol = np.real(K_eigs.max() * K.shape[0] * np.core.finfo(K.dtype).eps)
    lbb = np.max([rank_tol*2, rank_tol*2 - K_eigs.min()])
    alpha_range = np.geomspace(lbb, 100*K_eigs.max(), n_grid)

    if estimator_name in ['SVC', 'SVR']:
        param_dict = {'C': 1/(2*alpha_range)}
    else:
        param_dict = {'alpha': alpha_range}
    if verbose > 1:
        tol_scale = np.floor(np.log10(rank_tol))  # e.g. 2e-5 would be -5
        print("Hyperparameter grid selection:")
        print(f"rank_tol: {rank_tol}")
        print(f"tol_scale: 10^{tol_scale}")
        print(f"{param_dict}")
    return param_dict


# Function to convert model dictionary to kernel dictionary
def models_to_kernels(models_dict, w=None):
    kmat_with_params = {}
    for kname, params in models_dict.items():
        # Unweighted kernels
        if kname == 'linear':
            kmat_with_params[kname] = kj.kmat_linear
        if kname == 'rbf':
            for g in params['g']:
                kmat_with_params[f'{kname}_g_{g}'] = wrap(kj.kmat_rbf, g=g)
        if kname == 'generalized-js':
            for ab in params['ab']:
                # Note: generalized-js, only valid when a >= 1, 0.5 <= b <= a
                if ab[0] >= 1 and ab[1] >= 0.5 and ab[1] <= ab[0]:
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(
                        kj.kmat_hilbert1, a=ab[0], b=ab[1])
        if kname == 'hilbertian':
            for ab in params['ab']:
                # Note: hilbert, only valid when a >= 1 b <= -1 and
                # not both a, b are inf
                if ab[0] >= 1 and ab[1] <= -1 and (not jnp.isinf(ab[0])
                                                   or not jnp.isinf(ab[1])):
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(
                        kj.kmat_hilbert2, a=ab[0], b=ab[1])
        if kname == 'aitchison':
            for c in params['c']:
                kmat_with_params[f'{kname}_c_{c}'] = wrap(
                    kj.kmat_aitchison, c_X=c, c_Y=c)
        if kname == 'aitchison-rbf':
            for cg in params['cg']:
                kmat_with_params[f'{kname}_c_{cg[0]}_g_{cg[1]}'] = wrap(
                    kj.kmat_aitchison_rbf, c_X=cg[0], c_Y=cg[0], g=cg[1])
        if kname == 'heat-diffusion':
            for t in params['t']:
                kmat_with_params[f'{kname}_t_{t}'] = wrap(kj.kmat_hd, t=t)
        # Weighted kernels
        if kname == 'linear-weighted':
            kmat_with_params[f'{kname}'] = wrap(
                wkj.kmat_linear_weighted, w=w)
        if kname == 'rbf-weighted':
            for g in params['g']:
                kmat_with_params[f'{kname}_g_{g}'] = wrap(
                    wkj.kmat_rbf_weighted, g=g, w=w)
        if kname == 'generalized-js-weighted':
            for ab in params['ab']:
                # Note: generalized-js, only valid when a >= 1, 0.5 <= b <= a
                if ab[0] >= 1 and ab[1] >= 0.5 and ab[1] <= ab[0]:
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(
                        wkj.kmat_hilbert1_weighted, a=ab[0], b=ab[1],
                        w=w)
        if kname == 'hilbertian-weighted':
            for ab in params['ab']:
                # Note: hilbert, only valid when a >= 1 b <= -1 and
                # not both a, b are inf
                if ab[0] >= 1 and ab[1] <= -1 and (not jnp.isinf(ab[0])
                                                   or not jnp.isinf(ab[1])):
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(
                        wkj.kmat_hilbert2_weighted, a=ab[0], b=ab[1],
                        w=w)
        if kname == 'aitchison-weighted':
            for c in params['c']:
                kmat_with_params[f'{kname}_c_{c}'] = wrap(
                    wkj.kmat_aitchison_weighted, c=c, w=w)
        if kname == 'aitchison-rbf-weighted':
            for cg in params['cg']:
                kmat_with_params[f'{kname}_c_{cg[0]}_g_{cg[1]}'] = wrap(
                    wkj.kmat_aitchison_rbf_weighted, g=cg[1],
                    c=cg[0], w=w)
        if kname == 'heat-diffusion-weighted':
            for t in params['t']:
                kmat_with_params[f'{kname}_t_{t}'] = wrap(
                    wkj.kmat_hd_weighted, t=t, w=w)
    return kmat_with_params


# Function to sort and rank outer CV results
def top_model_per_kernel_class(kernel_dict, train_scores,
                               test_scores, selected_params):
    # Kernel classes in output order
    kernel_classes = ['aitchison', 'linear', 'generalized-js',
                      'hilbertian', 'rbf', 'aitchison-rbf', 'heat-diffusion']
    kernel_classes = kernel_classes + [kk + '-weighted'
                                       for kk in kernel_classes]
    estimator_keys = list(kernel_dict.keys())
    kmat_and_params = [ss.split('_', maxsplit=1) for ss in estimator_keys]
    # Add kernels that have non standard name to the end of ordering
    for kk in kmat_and_params:
        if kk[0] not in kernel_classes:
            kernel_classes.append(kk[0])
    # Adjust for kernels without parameters
    for kk, nn in enumerate(kmat_and_params):
        if len(nn) < 2:
            kmat_and_params[kk] = [nn[0], 'NA']
    # Aggregate results
    kmat_and_params = pd.DataFrame(kmat_and_params,
                                   columns=['kernel', 'kernel_params'])
    res = pd.DataFrame({'estimator_key': estimator_keys,
                        'kmat_fun': kernel_dict.values(),
                        'avg_train_score': train_scores.mean(axis=1),
                        'avg_test_score': test_scores.mean(axis=1),
                        'most_freq_best_param': [
                            Counter(row).most_common(1)[0][0]
                            for row in selected_params]
                        })
    res = pd.concat([kmat_and_params, res], axis=1)
    # Specify the kernel ordering and sort
    res['kernel'] = pd.Categorical(res['kernel'],
                                   kernel_classes)
    res.sort_values(['avg_test_score', 'avg_train_score', 'kernel'],
                    ascending=[False, False, True], inplace=True)
    # For each classes of kernels return best
    top_res = res.groupby('kernel')[
        ['estimator_key', 'kmat_fun',
         'avg_test_score', 'most_freq_best_param']].head(1)
    top_res.sort_values('avg_test_score', ascending=False, inplace=True)
    return top_res


# Internal function that fits a single model
def fit_single_model(X, y,
                     estimator,
                     kmat_fun,
                     scoring,
                     hyperpar_grid=None,
                     center_kmat=True,
                     return_gscv=False,
                     n_hyper_grid=10,
                     estimator_pars={},
                     n_fold=5,
                     n_jobs=6, verbose=0):
    """
    Fits a single model by selecting optimal hyperparameters
    based on CV
    """
    estimator_name = estimator.__class__.__name__
    estimator.set_params(**estimator_pars)

    # Select hyperparameter grid based on kernel
    if hyperpar_grid is None:
        hyperpar_grid = get_hyperpar_grid(
            X, kmat_fun, estimator, n_hyper_grid)
        if hyperpar_grid is None:
            return (None, np.inf)

    # Inner CV
    gscv = GridSearchCV(estimator, hyperpar_grid, cv=n_fold,
                        scoring=scoring, n_jobs=n_jobs, verbose=verbose)

    K = kmat_fun(X, X)
    if center_kmat:
        transformer = KernelCenterer().fit(K)
        K = transformer.transform(K)
    else:
        transformer = None

    # Add intercept for KernelRidge
    if estimator_name == "KernelRidge":
        intercept = np.mean(y)
    else:
        intercept = 0

    # Fit model
    gscv.fit(K, y-intercept)

    # Define prediction function
    if center_kmat:
        def pred_fun(X_new):
            K = kmat_fun(X_new, X)
            K = transformer.transform(K)
            return gscv.best_estimator_.predict(K) + intercept
    else:
        def pred_fun(X_new):
            K = kmat_fun(X_new, X)
            return gscv.best_estimator_.predict(K) + intercept

    # Training score
    train_score = gscv.score(K, y - intercept)

    # Collect output
    output = {'pred_fun': pred_fun,
              'intercept': intercept,
              'estimator': gscv.best_estimator_,
              'transformer': transformer,
              'train_score': train_score}
    if return_gscv:
        output['gscv'] = gscv

    return output


# Internal function that performs the outer CV
def outer_cv(X, y, estimator, kmat_fun, center_kmat,
             cv_split, scoring, hyperpar_grid,
             n_fold_outer, n_fold_inner, estimator_pars,
             n_hyper_grid, n_jobs, verbose=0):
    """
    Performs the outer CV
    """
    train_scores = np.full(n_fold_outer, np.NaN)
    test_scores = np.full(n_fold_outer, np.NaN)
    selected_params = np.full(n_fold_outer, np.NaN, dtype=object)

    # Select hyperparameter grid based on kernel
    if hyperpar_grid is None:
        hyperpar_grid = get_hyperpar_grid(
            X, kmat_fun, estimator, n_hyper_grid)
        if hyperpar_grid is None:
            return {'train_scores': np.NaN,
                    'test_scores': np.NaN,
                    'selected_params': np.NaN}

    for kk, (train_index, test_index) in enumerate(cv_split):
        if verbose > 1 and kk % 1 == 0:
            print(f"fold {kk+1} out of {n_fold_outer} folds")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Fit model
        fitted_mod = fit_single_model(
            X_train, y_train, estimator,
            kmat_fun,
            scoring,
            hyperpar_grid=hyperpar_grid,
            center_kmat=center_kmat,
            n_fold=n_fold_inner,
            return_gscv=True,
            estimator_pars=estimator_pars,
            n_jobs=n_jobs, verbose=verbose)
        # training score
        train_scores[kk] = fitted_mod['train_score']
        # test scorei
        Ktest = kmat_fun(X_test, X_train)
        if center_kmat:
            Ktest = fitted_mod['transformer'].transform(Ktest)
        test_scores[kk] = fitted_mod['gscv'].score(
            Ktest, y_test-fitted_mod['intercept'])
        # best parameters
        selected_params[kk] = vars(fitted_mod['gscv'].best_estimator_)[
            list(hyperpar_grid.keys())[0]]

    return {'train_scores': train_scores,
            'test_scores': test_scores,
            'selected_params': selected_params}
