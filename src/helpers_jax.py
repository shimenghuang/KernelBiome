from functools import partial
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
