# %%
# load libs
# ^^^^^^
from pathlib import Path  # nopep8
import sys  # nopep8
path_root = Path(__file__).parents[1]  # nopep8
sys.path.append(str(path_root))  # nopep8
print(sys.path)  # nopep8

from os.path import join
from kernelbiome.cfi_and_cpd import *
from kernelbiome.utils_cv import *
from kernelbiome.utils_result import *
from kernelbiome.metrics_jax import *
from kernelbiome.kernels_jax import *
import numpy as np
import pandas as pd
from jax import vmap


# %%
# set seed
# ^^^^^^
use_seed = int(sys.argv[1])
print(f"seed: {use_seed}")
rng = np.random.default_rng(use_seed)

# file_path = "/Users/hrt620/Documents/projects/kernelbiome_proj/kernelbiome/src/output_dgp6_kr_local/"  # local path
file_path = "output/"  # server
# %%
# global vars
# ^^^^^^^^^^
p = 9
ref_pt = np.array([0.06544714, 0.08760064, 0.17203408, 0.07502236, 0.1642615,
                   0.03761901, 0.18255478, 0.13099514, 0.08446536])
print(ref_pt)

# %%
# helper funs
# ^^^^^^^^^^


def true_fun_(x):
    return 100*k_hilbert2(x, ref_pt, a=1, b=-np.inf)


def true_fun(X):
    return vmap(true_fun_)(X)


def gen_data(n):
    mu = [0, 0, 0]
    cov = [[1, 0.25, -0.25],
           [0.25, 1, 0.25],
           [-0.25, 0.25, 1]]
    X1 = rng.multivariate_normal(mu, cov, n)  # 0, 1, 2
    X2 = rng.multivariate_normal(mu, cov, n)  # 3, 4, 5
    X3 = rng.multivariate_normal(mu, cov, n)  # 6, 7, 8
    X = np.concatenate([X1, X2, X3], axis=1)
    X = np.exp(X)
    X /= X.sum(axis=1)[:, None]
    y = true_fun(X) + rng.normal(0, 1, n)
    return X, y


# %%
# calculate true CPD
# ^^^^^^^^^^
n = 10**3
p = 9
n_grid = 100
cpd_vals_true = np.zeros((p, n_grid))
supp_grid = get_gen_grid(n_grid, p, 0.001, 0.999)
X, y = gen_data(n)
cpd_vals_true = get_cfi(X, supp_grid, true_fun)
np.save(join(file_path, f"kr_cpd_true_seed_{use_seed}.npy"), cpd_vals_true)


# %%
# estimator and main function
# ^^^^^^^^^^
def my_kb_estimator(X, y):
    """
    For getting consistancy result purpose.
    """
    n_fold_outer = 10
    n_fold_inner = 5
    scoring = 'neg_mean_squared_error'
    n_jobs = -1
    param_grid_kr = dict(alpha=[1/10**x for x in [-1, 0, 1, 2, 3]])
    kernel_params_dict = default_kernel_params_grid()
    kmat_with_params = get_kmat_with_params(kernel_params_dict)
    train_scores_all, test_scores_all, selected_params_all = run_experiments(X, y, kmat_with_params, param_grid_kr, None, None, center_kmat=True,
                                                                             n_fold_outer=n_fold_outer, n_fold_inner=n_fold_inner, type='regression', scoring='neg_mean_squared_error', kernel_estimator='kr',
                                                                             n_jobs=n_jobs, random_state=rng, verbose=0)
    best_kernel_models = top_models_in_each_group(
        kmat_with_params, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=True)
    model = best_kernel_models.iloc[0]
    pred_fun, gscv = refit_best_model(
        X, y, 'KernelRidge', param_grid_kr, model, scoring)
    return pred_fun, gscv, model.kmat_fun, model.estimator_key


def run_consistency(ns, gen_data_fun, estimator_fun, n_repeat=10):
    """
    ns: List
        a list of number of samples.
    gen_data_fun: Callable
        a function that takes only the number of samples n and returns X, y
    estimator: Callable
        a function that takes X, y and returns a prediction function, the selected hyperparameters, kernel matrix function, and the name of the kernel
    """
    cpd_est_list = []
    for ii, n in enumerate(ns):
        # clear_output(wait=True)
        print(f'-- ii = {ii} (n = {n}) --')
        cpd_loc = np.zeros((n_repeat, p, n_grid))
        for jj in range(n_repeat):
            X, y = gen_data_fun(n)
            pred_fun, gscv, kmat_fun, kernel_args_str = estimator_fun(X, y)
            cpd_loc[jj] = get_cfi(X, supp_grid, pred_fun)
        cpd_est_list.append(cpd_loc)
    # average MSE of the 9 curves at each sample size n
    mse = np.zeros((len(ns), n_repeat))
    for ii, n in enumerate(ns):
        for jj in range(n_repeat):
            # MSE of each curve, shape is (p,)
            mse_loc = np.mean((cpd_vals_true-cpd_est_list[ii][jj])**2, axis=1)
            mse[ii, jj] = np.mean(mse_loc)
    np.save(join(file_path, f"kr_cpd_est_seed_{use_seed}.npy"), cpd_est_list)
    return pd.DataFrame(mse.T, columns=ns)


# %%
# run main
# ^^^^^^^^^^
ns = [50, 100, 200, 500]
n_repeat = 1
mse_df = run_consistency(ns, gen_data, my_kb_estimator, n_repeat)
mse_df.to_csv(
    join(file_path, f"kr_cpd_mse_df_seed_{use_seed}.csv"), index=False)

# %%
