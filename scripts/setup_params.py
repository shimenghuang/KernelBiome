# %%
# load libs
# ^^^^^^

import sys  # nopep8
sys.path.insert(0, "../")  # nopep8

from os.path import join
from src.kernels_jax import *
from src.utils import *

# %%
# median heruisic for RBF and Aitchison-RBF
# ^^^^^^


def median_heruistic(X_df):

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

    return g1, g2

# %%
# 1) KB estimator
# ^^^^^^


def kb_params(g1, g2):

    kernel_params_dict = default_kernel_params_grid(g1, g2)
    kmat_with_params = get_kmat_with_params(kernel_params_dict)
    print(f"number of kernels: {len(kmat_with_params)}")

    return kmat_with_params

# %%
# WKB estimator
# ^^^^^^


def wkb_params(g1, g2, w_unifrac_ma, w_unifrac_mb):

    # UniFrac weight with MA
    weighted_kernel_params_dict_ma = default_weighted_kernel_params_grid(
        w_unifrac_ma, g1, g2)
    weighted_kmat_with_params_ma = get_weighted_kmat_with_params(
        weighted_kernel_params_dict_ma, w_unifrac=w_unifrac_ma)

    # UniFrac weight with MB
    weighted_kernel_params_dict_mb = default_weighted_kernel_params_grid(
        w_unifrac_mb, g1, g2)
    weighted_kmat_with_params_mb = get_weighted_kmat_with_params(
        weighted_kernel_params_dict_mb, w_unifrac=w_unifrac_mb)

    return weighted_kmat_with_params_ma, weighted_kmat_with_params_mb

# %%
# main
# ^^^^^^^


def main(X_df, w_unifrac_ma, w_unifrac_mb):
    g1, g2 = median_heruistic(X_df)
    kmat_with_params = kb_params(g1, g2)
    weighted_kmat_with_params_ma, weighted_kmat_with_params_mb = wkb_params(
        g1, g2, w_unifrac_ma, w_unifrac_mb)
    return kmat_with_params, weighted_kmat_with_params_ma, weighted_kmat_with_params_mb
