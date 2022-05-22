import numpy as np
import pandas as pd


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


def aggregate_level(X_count, comp_lbl, levels, start_levels=["kingdom", "phylum", "class", "order", "family", "genus", "species"], start_delim=";"):
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
