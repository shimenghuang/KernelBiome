import numpy as np
from jax import grad
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from .utils_compositional import C_partial

# ---- CPD calculation ----


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


def get_cpd(X, supp_grid, pred_fun, comp_idx=None, rescale=True, verbose=False):
    """
    Calculate CPD vales based on a fitted prediction function.

    Parameters:
    supp_grid: np.array of shape (n_grid, p)
        The j-th column contains values within the support range of j-th column of X.
    pred_fun: callable
        Could be either a fitted model or the true underlying function that takes only X as the argument.
    comp_idx: np.array
        Indices of components to calculate CPD for. If None, calculate CPD for all components.
    rescale: bool
        Whether to partially rescale the vector to have summation one. Default True which corresponds to the correct CPD definition.

    Returns:
        np.array of shape (len(cpd_idx), supp_grid.shape[0])
    """
    assert(X.shape[1] == supp_grid.shape[1])
    n, p = X.shape
    comp_idx = range(p) if comp_idx is None else comp_idx
    n_grid = supp_grid.shape[0]
    cpd_vals = np.zeros((len(comp_idx), n_grid))
    for jj, idx in enumerate(comp_idx):
        if verbose:
            print(f'-- jj: {jj}: comp {idx} --')
        for ii in range(n):
            if verbose and ii % 10 == 0:
                print(f'  ii: {ii}')
            G = np.repeat(X[ii][None, :], n_grid, axis=0)
            G[:, idx] = supp_grid[:, idx].copy()
            G = C_partial(G, idx) if rescale else G
            cpd_vals[jj] += pred_fun(G)
    cpd_vals /= n
    cpd_vals -= np.mean(pred_fun(X))
    return cpd_vals


# ---- CFI calculation ----

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
    Calculate the derivative of the pertubation function 

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
    """
    Calculate the derivative of the pertubation function without rescaling

    .. math:: \phi_{\text{no proj}}^j(x,c) = (x^1,\cdots,cx^j,\cdots,x^p)

    w.r.t. c at c=1.
    """
    X_new = np.zeros_like(X)
    X_new[:, jj] = X[:, jj].copy()
    return X_new


def df_ke_dual_mat(X_eval, X_fit, dual_coef, idx_supp, k_fun, **kwargs):
    """
    Calculate the derivative of a fitted kernel function via the dual problem.
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


def get_cfi(X, df, proj=True):
    """
    Calculate CFI based on a fitted estimator and the observations.
    """
    n, p = X.shape
    res = np.zeros(p)
    dphi_fun = dphi_mat if proj else dphi_no_proj_mat
    for jj in range(p):
        res[jj] = np.mean(np.sum(df * dphi_fun(X, jj), axis=1))
    return res


# ---- plotting functions ----


def plot_cpd(supp_grid, cfi_vals, labels=None, colors=None, axs=None, xlabel=r'$X^{j}$', ylable=r'CPD'):
    """
    Plot CDP curves.
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


def plot_cfi(cfi_index_vals, fmt='%.2f', labels=None, colors=None, axs=None, ascending=False):
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


# ---- for comparison ----


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
