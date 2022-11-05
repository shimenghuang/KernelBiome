# %%
# libs
# ##

from pathlib import Path  # nopep8
import sys  # nopep8
path_root = Path(__file__).parents[2]  # nopep8
sys.path.append(str(path_root))  # nopep8
print(sys.path)  # nopep8

import pandas as pd
import numpy as np
from jax import grad, vmap
from kernelbiome.kernelbiome import KernelBiome
from kernelbiome.helpers_analysis import (get_cfi, get_cpd, get_gen_grid)
from kernelbiome.kernels_jax import k_hilbert2
import argparse
from os.path import join
import os
print(f'cwd: {os.getcwd()}')


# %%
# paths
# ##

if os.getcwd().endswith('consistency'):
    # cwd is current file's location
    output_path = "./results/"
else:
    # assuming running from kernelbiome_tmp
    output_path = "experiments/consistency/results/"
os.makedirs(output_path, exist_ok=True)

# %%
# helper funs
# ##

parser = argparse.ArgumentParser(
    description="Run consistency experiment",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-s", "--seed", type=int, help="seed number")
args = parser.parse_args()
config = vars(args)
print(config)
use_seed = config['seed']

# use_seed = 0  # int(sys.argv[1])

print(f"seed: {use_seed}")
rng = np.random.default_rng(use_seed)

p = 9
ref_pt = np.array([0.06544714, 0.08760064, 0.17203408, 0.07502236, 0.1642615,
                   0.03761901, 0.18255478, 0.13099514, 0.08446536])
n_grid = 100
supp_grid = get_gen_grid(n_grid, p, 0.001, 0.999)


def true_fun_(x):
    return 100*k_hilbert2(x, ref_pt, a=1, b=-np.inf)


def true_fun(X):
    return vmap(true_fun_)(X)


# derivative of true_fun_
d_true_fun_ = grad(true_fun_)


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


def my_kb_estimator(X, y):

    KB = KernelBiome(kernel_estimator='KernelRidge',
                     center_kmat=True,
                     hyperpar_grid=None,
                     models=None,
                     cv_pars={},
                     estimator_pars={},
                     n_jobs=6,
                     random_state=use_seed,
                     verbose=1)
    KB.fit(X, y)
    return KB


def run_consistency(ns, gen_data_fun, estimator_fun, n_repeat=10):
    """
    ns: List
        a list of number of samples.
    gen_data_fun: Callable
        a function that takes only the number of samples n and returns X, y
    estimator: Callable
        a function that takes X, y and returns a prediction function, the selected hyperparameters, kernel matrix function, and the name of the kernel
    """
    cfi_true_list = []
    cfi_est_list = []
    cpd_true_list = []
    cpd_est_list = []

    for ii, n in enumerate(ns):
        print(f'-- ii = {ii} (n = {n}) --')
        cfi_true = np.zeros((n_repeat, p))
        cfi_est = np.zeros((n_repeat, p))
        cpd_true = np.zeros((n_repeat, p, n_grid))
        cpd_est = np.zeros((n_repeat, p, n_grid))
        for jj in range(n_repeat):
            print(f'-- * jj = {jj+1} (out of {n_repeat}) --')
            X, y = gen_data_fun(n)
            # calculate true CFI and CPD
            df_true = vmap(d_true_fun_)(X)
            cfi_true[jj] = get_cfi(X, df_true)
            cpd_true[jj] = get_cpd(X, supp_grid, true_fun)
            # estimate CFI and CPD
            KB = estimator_fun(X, y)
            cfi_est[jj] = KB.compute_cfi(X)
            cpd_est[jj] = KB.compute_cpd(X, supp_grid)

        cfi_true_list.append(cfi_true)
        cfi_est_list.append(cfi_est)
        cpd_true_list.append(cpd_true)
        cpd_est_list.append(cpd_est)

        # save results in every iteration
        np.save(
            join(output_path, f"cfi_true_seed_{use_seed}.npy"), cfi_true_list)
        np.save(
            join(output_path, f"cpd_true_seed_{use_seed}.npy"), cpd_true_list)
        np.save(
            join(output_path, f"cfi_est_seed_{use_seed}.npy"), cfi_est_list)
        np.save(
            join(output_path, f"cpd_est_seed_{use_seed}.npy"), cpd_est_list)

    # calculate MSE
    mse_cfi = np.zeros((len(ns), n_repeat))
    mse_cpd = np.zeros((len(ns), n_repeat))
    for ii, n in enumerate(ns):
        for jj in range(n_repeat):
            mse_cfi[ii, jj] = np.mean(
                (cfi_true_list[ii][jj]-cfi_est_list[ii][jj])**2)
            # average MSE of the 9 curves at each sample size n, MSE of each curve, shape is (p,)
            mse_loc = np.mean(
                (cpd_true_list[ii][jj]-cpd_est_list[ii][jj])**2, axis=1)
            mse_cpd[ii, jj] = np.mean(mse_loc)

    return pd.DataFrame(mse_cfi.T, columns=ns), pd.DataFrame(mse_cpd.T, columns=ns)


# %%
# run main
# ^^^^^^^^^^
ns = [50, 100, 200, 500]
n_repeat = 1
mse_cfi_df, mse_cpd_df = run_consistency(
    ns, gen_data, my_kb_estimator, n_repeat)
mse_cfi_df.to_csv(
    join(output_path, f"cfi_mse_seed_{use_seed}.csv"), index=False)
mse_cpd_df.to_csv(
    join(output_path, f"cpd_mse_seed_{use_seed}.csv"), index=False)

# %%
