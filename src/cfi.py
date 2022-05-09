import numpy as np
from jax import grad, vmap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from src.utils import C_fun, C_partial

# ---- others ----


def permutation_importance(X, y, pred_fun, B=50, scoring='neg_mean_squared_error', rng=None):
    rng = rng or np.random.default_rng()
    perm_imp = np.zeros(X.shape[1])
    if scoring == "neg_mean_squared_error":
        def score_fun(y, yhat): return -mean_squared_error(y, yhat)
    elif scoring == "accuracy":
        def score_fun(y, yhat): return np.mean(yhat == y)
    else:
        raise ValueError(f"scoring {scoring} not implemented.")
    baseline_score = score_fun(y, pred_fun(X))
    # print(f"baseline score: {baseline_score}")
    for jj in range(X.shape[1]):
        # print(f"-- jj: {jj} --")
        for ii in range(B):
            # print(f"  ii: {ii}")
            X_loc = X.copy()
            rng.shuffle(X_loc[:, jj])
            # print(f"local score: {score_fun(y, pred_fun(X_loc))}")
            perm_imp[jj] += baseline_score - score_fun(y, pred_fun(X_loc))
    perm_imp /= B
    return perm_imp


# ---- general helpers ----


def get_gen_grid(n_grid, p, min_val, max_val):
    """
    Get general grid of shape (n_grid,p).

    Each column contains uniform grid values in [min_val, max_val).
    If `min_val` or `max_val` is array, the length should be `p`.
    """
    if np.isscalar(min_val):
        min_val = np.repeat(min_val, p)
    if np.isscalar(max_val):
        max_val = np.repeat(max_val, p)
    assert(len(min_val) == len(max_val))
    gen_grid = np.linspace(min_val, max_val, num=n_grid)
    return gen_grid


def get_supp_grid(X, n_grid):
    """
    Construct support grid based on min and max values in each column of X.
    """
    p = X.shape[1]
    supp_grid = np.zeros((n_grid, p))
    for jj in range(p):
        supp_grid[:, jj] = np.linspace(
            min(X[:, jj]), max(X[:, jj]), num=n_grid)
    return supp_grid


# ---- CFI calculations (do-intervention approach) ----


def get_cfi(X, supp_grid, pred_fun, rescale=True, verbose=False):
    """
    Calculate CFI vales for a kernel based or non-kernel based function.

    supp_grid: np.array of shape (n_grid, p)
        Its j-th column contains values within the support range of j-th column of X.
    pred_fun: callable
        Could be either a fitted model or the true underlying function that takes only X as the argument.
    """
    assert(X.shape[1] == supp_grid.shape[1])
    n, p = X.shape
    n_grid = supp_grid.shape[0]
    cfi_vals = np.zeros((p, n_grid))
    for jj in range(p):
        if verbose:
            print(f'jj: {jj}')
        for ii in range(n):
            G = np.repeat(X[ii][None, :], n_grid, axis=0)
            G[:, jj] = supp_grid[:, jj].copy()
            G = C_partial(G, jj) if rescale else G
            cfi_vals[jj] += pred_fun(G)
    cfi_vals /= n
    cfi_vals -= np.mean(pred_fun(X))
    return cfi_vals


def get_cfi_new(X, pert_grid, pred_fun, kmat_fun=None, verbose=False, *args):
    """
    Calculate CFI vales for a kernel based or non-kernel based function.

    supp_grid: np.array of shape (n_grid, p)
        Its j-th column contains values within the support range of j-th column of X.
    pred_fun: callable
        Could be either a fitted model or the true underlying function.
    kmat_fun: callable or None
        The function that calculates the kernel matrix if the fitted model is kernel based.
    *args: additional arguments to `kmat_fun`
    """
    assert(X.shape[1] == pert_grid.shape[1])
    n, p = X.shape
    n_grid = pert_grid.shape[0]
    cfi_vals = np.zeros((p, n_grid))
    for jj in range(p):
        if verbose:
            print(f'jj: {jj}')
        for ii in range(n):
            G = np.repeat(X[ii][None, :], n_grid, axis=0)
            # pertubation on jj-th component for ii-th observation
            G[:, jj] *= pert_grid[:, jj]
            G = C_fun(G)
            if kmat_fun is None:
                cfi_vals[jj] += pred_fun(G)
            else:
                K = kmat_fun(G, X, *args)
                cfi_vals[jj] += pred_fun(K)
    cfi_vals /= n
    if kmat_fun is None:
        cfi_vals -= np.mean(pred_fun(X))
    else:
        K = kmat_fun(X, X, *args)
        cfi_vals -= np.mean(pred_fun(K))
    return cfi_vals


def get_cfi_index(X, pred_fun, rescale=True, verbose=False):
    p = X.shape[1]
    cfi_vals = get_cfi(X, X, pred_fun, rescale=rescale, verbose=verbose)
    cfi_index_vals = np.zeros(p)
    for ii in range(X.shape[1]):
        k_max = np.argmax(cfi_vals[ii])
        k_min = np.argmin(cfi_vals[ii])
        cfi_index_vals[ii] = (
            cfi_vals[ii, k_max] - cfi_vals[ii, k_min]) * np.sign(X[k_max, ii] - X[k_min, ii])
    return cfi_index_vals


# def get_cfi_index(X, cfi_vals):
#     p = cfi_vals.shape[0]
#     cfi_index_vals = np.zeros(p)
#     for ii in range(X.shape[1]):
#         k_max = np.argmax(cfi_vals[ii])
#         k_min = np.argmin(cfi_vals[ii])
#         cfi_index_vals[ii] = (
#             cfi_vals[ii, k_max] - cfi_vals[ii, k_min]) * np.sign(X[k_max, ii] - X[k_min, ii])
#     return cfi_index_vals


# ---- CFI-index based on pertubation approach ----


def num_df_mat(fun, h, X):
    n, p = X.shape
    res = np.zeros((n, p))
    for jj in range(p):
        ej = np.zeros(p)
        ej[jj] = 1
        res[:, jj] = (fun(X + h/2*ej) - fun(X - h/2*ej))/h
    return res


def dphi_mat(X, jj):
    """
    Take derivative of the pertubation function

    .. math:: \phi^j(x,c) = C(x^1,\cdots,cx^j,\cdots,x^p)

    w.r.t. c at c=1.
    """
    X_new = X.copy()
    n, p = X.shape
    Xj = X_new[:, jj].copy()[:, None]
    X_new = -np.tile(Xj, p) * X_new
    X_new[:, jj] += Xj[:, 0]
    return X_new


def dphi_no_proj_mat(X, jj):
    X_new = np.zeros_like(X)
    X_new[:, jj] = X[:, jj].copy()
    return X_new


def df_ke_dual_mat(X_eval, X_fit, dual_coef, idx_supp, k_fun, **kwargs):
    """
    Take derivative of kernel estimator function via the dual problem.
    """
    assert(len(dual_coef) == len(idx_supp))
    n, p = X_eval.shape
    df = np.zeros((n, p))
    dy_k_fun = grad(k_fun, argnums=1)
    for ii in range(n):
        for jj in range(len(dual_coef)):
            dk = np.array(dy_k_fun(X_fit[idx_supp[jj]], X_eval[ii], **kwargs))
            # Note: now set all nan's to 0.0
            dk = np.nan_to_num(dk, False, nan=0.0, posinf=0.0, neginf=0.0)
            df[ii] += dual_coef[jj] * dk
    return df

# def df_ke_dual_mat(X, dual_coef, idx_supp, k_fun, **kwargs):
#     """
#     Take derivative of kernel estimator function via the dual problem.
#     """
#     assert(len(dual_coef) == len(idx_supp))
#     n, p = X.shape
#     df = np.zeros((n, p))
#     dy_k_fun = grad(k_fun, argnums=1)
#     for ii in range(n):
#         for jj in range(len(dual_coef)):
#             dk = np.array(dy_k_fun(X[idx_supp[jj]], X[ii], **kwargs))
#             # Note: now set all nan's to 0.0
#             dk = np.nan_to_num(dk, False, nan=0.0, posinf=0.0, neginf=0.0)
#             df[ii] += dual_coef[jj] * dk
#     return df


def get_perturbation_cfi_index(X, df, proj=True):
    n, p = X.shape
    res = np.zeros(p)
    dphi_fun = dphi_mat if proj else dphi_no_proj_mat
    for jj in range(p):
        res[jj] = np.mean(np.sum(df * dphi_fun(X, jj), axis=1))
    return res


# ---- plotting functions ----


def plot_cfi(supp_grid, cfi_vals, labels=None, colors=None, axs=None, xlabel=r'$X^{j}$', ylable=r'CPD'):
    """
    Plot CFI as curves.
    """
    if axs is None:
        fig, axs = plt.subplots()
    # axs.set_xlim(0, 1)
    for ii in range(cfi_vals.shape[0]):
        color = colors[ii] if colors is not None else None
        lbl_ii = '' if labels is None else labels[ii]
        idx = np.argsort(supp_grid[:, ii])
        axs.plot(supp_grid[idx, ii], cfi_vals[ii][idx], color=color,
                 label=lbl_ii, alpha=0.6)  # f'$x^{ii+1}$'
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['left'].set(color='gray', linewidth=2, alpha=0.2)
    axs.spines['bottom'].set(color='gray', linewidth=2, alpha=0.2)
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylable)
    return axs


def plot_cfi_index(cfi_index_vals, fmt='%.2f', labels=None, colors=None, axs=None, ascending=False):
    """
    Plot CFI-index bars.
    """
    if axs is None:
        fig, axs = plt.subplots()
    n_comp = cfi_index_vals.shape[0]
    axs.axvline(0, ymin=-1, ymax=n_comp,
                color='gray', linewidth=2, alpha=0.2)
    pad = max(np.abs(cfi_index_vals)) * 0.2
    label_x = max(cfi_index_vals) + pad
    pos = np.argsort(np.abs(cfi_index_vals)
                     ) if ascending else np.argsort(-np.abs(cfi_index_vals))
    colors_use = colors[:n_comp] if colors is not None else None
    for ii in range(n_comp):
        y = ii if ascending else n_comp-ii-1
        color = colors_use[pos[ii]] if colors is not None else None
        p = axs.barh(y, cfi_index_vals[pos[ii]], height=0.8,
                     label=f'$X^{pos[ii]+1}$', align='center',
                     alpha=0.6, color=color)
        axs.bar_label(p, fmt=fmt, label_type='center', size='medium')
        lbl_ii = '' if labels is None else labels[pos[ii]]
        axs.text(label_x, y, lbl_ii)
    axs.yaxis.set_visible(False)
    plt.setp(axs.spines.values(), visible=False)
    axs.tick_params(bottom=False, labelbottom=False)
    axs.set_title(r'CFI', y=-0.2)
    return axs


# # ---- old implementations ----

# def CFI_KR(X, kernel_fun, pred_fun, verbose=False, *args):
#     """
#     Calculate CFI under kernel ridge regression.

#     This is due to that sklearn's KernelRidge only accept precomputed kernel matrix or pairwise callable function as kernel.
#     Here precomputed kernel is used.
#     """
#     n, p = X.shape
#     influence_all = np.zeros((p, n))
#     for jj in range(p):
#         if verbose:
#             print(f'jj: {jj}')
#         for ii in range(n):
#             G = np.repeat(X[ii][None, :], n, axis=0)
#             # K_base = kernel_fun(G, X, *args)
#             G[:, jj] = X[:, jj].copy()
#             G = C_partial(G, jj)
#             K = kernel_fun(G, X, *args)
#             influence_all[jj] += pred_fun(K)
#             # influence_all[jj] += pred_fun(K) - pred_fun(K_base)
#     influence_all /= n
#     K = kernel_fun(X, X, *args)
#     influence_all -= np.mean(pred_fun(K))
#     return influence_all


# def CFI(X, fun, verbose=False):
#     """
#     Calculate CFI under true function or fitted SVR predictor.
#     """
#     n, p = X.shape
#     influence_all = np.zeros((p, n))
#     for jj in range(p):
#         if verbose:
#             print(f'jj: {jj}')
#         for ii in range(n):
#             # repeat each X[ii] n times and only replace jj-th column
#             G = np.repeat(X[ii][None, :], n, axis=0)
#             G[:, jj] = X[:, jj].copy()
#             G = C_partial(G, jj)
#             influence_all[jj] += fun(G)
#     influence_all /= n
#     influence_all -= np.mean(fun(X))
#     return influence_all


# def CFI_plot(X, cfi_vals, axs=None):
#     """
#     cfi_vals: np.array of shape (p,n)
#     """
#     if axs is None:
#         fig, axs = plt.subplots(1, 1, figsize=(16, 6))
#     for ii in range(cfi_vals.shape[0]):
#         axs.scatter(X[:, ii], cfi_vals[ii], label=ii, s=15)
#     axs.legend(bbox_to_anchor=(1, 1))
#     return axs


# def cfi_plot(X, influence_all, labels=None, colors=None, axs=None):
#     if axs is None:
#         fig, axs = plt.subplots()
#     # axs.set_xlim(0, 1)
#     for ii in range(influence_all.shape[0]):
#         color = colors[ii] if colors is not None else None
#         lbl_ii = '' if labels is None else labels[ii]
#         axs.scatter(X[:, ii], influence_all[ii], color=color,
#                     label=lbl_ii, s=5, alpha=0.6)  # f'$x^{ii+1}$'
#     axs.spines['right'].set_visible(False)
#     axs.spines['top'].set_visible(False)
#     axs.spines['left'].set(color='gray', linewidth=2, alpha=0.2)
#     axs.spines['bottom'].set(color='gray', linewidth=2, alpha=0.2)
#     axs.set_xlabel(r'$X^{j}$')
#     axs.set_ylabel(r'CFI')
#     return axs


# def cfii_plot(cfii_vals, fmt='%.2f', labels=None, colors=None, axs=None, ascending=False):

#     if axs is None:
#         fig, axs = plt.subplots()
#     n_comp = cfii_vals.shape[0]
#     axs.axvline(0, ymin=-1, ymax=n_comp, color='gray', linewidth=2, alpha=0.2)
#     pad = max(np.abs(cfii_vals)) * 0.2
#     label_x = max(cfii_vals) + pad
#     # axs.set_xlim(min(cfii_vals)-pad, max(cfii_vals)+pad)
#     # axs.set_ylim(-1, n_comp)
#     pos = np.argsort(np.abs(cfii_vals)
#                      ) if ascending else np.argsort(-np.abs(cfii_vals))
#     colors_use = colors[:n_comp] if colors is not None else None
#     for ii in range(n_comp):
#         y = ii if ascending else n_comp-ii-1
#         color = colors_use[pos[ii]] if colors is not None else None
#         p = axs.barh(y, cfii_vals[pos[ii]], height=0.8,
#                      label=f'$X^{pos[ii]+1}$', align='center',
#                      alpha=0.6, color=color)
#         # axs.bar_label(
#         #     p, labels=[f'$I^{ii} = {np.round(cfii_vals[ii], n_digits)}$'], label_type='center', size='medium')
#         axs.bar_label(p, fmt=fmt, label_type='center', size='medium')
#         lbl_ii = '' if labels is None else labels[pos[ii]]
#         axs.text(label_x, y, lbl_ii)
#         # axs.bar_label(p, labels=[lbl_ii],
#         #               label_type='edge', size='medium', padding=pad)
#     axs.yaxis.set_visible(False)
#     plt.setp(axs.spines.values(), visible=False)
#     axs.tick_params(bottom=False, labelbottom=False)
#     axs.set_xlabel(r'CFI-index')
#     # axs.set_title('Compositional feature influence index', y=-0.1)
#     return axs
