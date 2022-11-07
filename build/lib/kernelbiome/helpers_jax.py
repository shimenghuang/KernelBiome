from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax import vmap


def wrap(fun, **kwargs):
    """
    e.g. wrap(kmat_hilbert1, a=1, b=0.5) which is a function taking X, Y only
    """
    def calc(X, Y):
        return fun(X, Y, **kwargs)
    return calc


@partial(jit, static_argnames=('axis'), inline=True)
def gmean(x, axis=0):
    """Compute the geometric mean along the specified axis.

    Return the geometric average of the array elements.
    That is:  n-th root of (x1 * x2 * ... * xn).

    Parameters
    ----------
    x : jax.numpy.ndarray

    Returns
    -------
    gmean : jax.numpy.ndarray

    """
    log_x = jnp.log(x)
    return jnp.exp(jnp.mean(log_x, axis=axis))


@jit
def squared_euclidean_distances(X, Y):
    XX = jnp.einsum("ij,ij->i", X, X)[:, jnp.newaxis]
    YY = jnp.einsum("ij,ij->i", Y, Y)[jnp.newaxis, :]
    distances = -2 * X.dot(Y.T)
    distances += XX
    distances += YY
    distances = jnp.maximum(distances, 0)
    return distances


def gram(func, X, Y, **kwargs):
    """Computes the gram matrix.

    Given a bivariate function and additional keyword arguments,
    the gram matrix is calculated via `jax.vmap` between points
    in `X` and `Y`.

    Parameters
    ----------
    func : Callable
        a callable function (kernel or distance)
    X : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    Y : jax.numpy.ndarray
        other input dataset (n_samples, n_features)

    Returns
    -------
    mat : jax.numpy.ndarray
        the gram matrix.

    Examples
    --------

    >>> gram(k_rbf, X, Y, g=1)
    """
    return vmap(lambda x: vmap(lambda y: func(x, y, **kwargs))(Y))(X)


def rbf_median_heruistic(X, clr=False, eps=0):
    """
    X: jax.numpy.ndarray
    clr: whether to central-log-ratio transform X first.
    eps: if clr is True, may require eps to be added to X to ensure positivity.
    """

    Xc = X + eps
    if clr:
        gm_Xc = gmean(Xc, axis=1)
        clr_Xc = np.log(Xc/gm_Xc[:, None])
        M = squared_euclidean_distances(clr_Xc, clr_Xc)
    else:
        M = squared_euclidean_distances(Xc, Xc)

    M_triu = M[np.triu_indices(n=M.shape[0], k=1, m=M.shape[1])]
    g = 1.0/np.median(M_triu)
    return g
