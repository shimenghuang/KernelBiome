from functools import partial
import jax.numpy as jnp
from jax import jit
from .helpers_jax import *


# ---- kernel functions on vectors ----

@jit
def k_linear(x, y):
    """Linear kernel

    .. math:: k_i = \sum_i^N x_i*y_i

    Parameters
    ----------
    x : jax.numpy.ndarry
        input arry of shape (n_features,)
    y : jax.numpy.ndarry
        input arry of shape (n_features,)

    Returns
    -------
    k_val : jax.numpy.ndarray
    """
    inv_p = 1.0/x.shape[0]
    return x.dot(y) - inv_p


@partial(jit, static_argnames=('g'), inline=True)
def k_rbf(x, y, g):
    """Radial Basis Function (RBF) kernel.

    .. math::

        k(\mathbf{x,y}) = \\
           \\exp \left( - \\gamma\\
           ||\\mathbf{x} - \\mathbf{y}||^2_2\\
            \\right)


    Parameters
    ----------
    x : jax.numpy.ndarry
        input arry of shape (n_features,)
    y : jax.numpy.ndarry
        input arry of shape (n_features,)
    g : float
        bandwidth parameter

    Returns
    -------
    k_val : jax.numpy.ndarray
    """
    return jnp.exp(-g*((x-y)**2).sum())


@partial(jit, static_argnames=('c'), inline=True)
def k_aitchison(x, y, c=0):
    """Aitchison kernel.

    .. math::

        # TODO


    Parameters
    ----------
    x : jax.numpy.ndarry
        input arry of shape (n_features,)
    y : jax.numpy.ndarry
        input arry of shape (n_features,)
    c : float
        pseudo-count parameter

    Returns
    -------
    k_val : jax.numpy.ndarray
    """
    x = x + c
    x = x / x.sum()
    y = y + c
    y = y / y.sum()
    gm_x = gmean(x)
    gm_y = gmean(y)
    clr_x = jnp.log(x/gm_x)
    clr_y = jnp.log(y/gm_y)
    return (clr_x*clr_y).sum()


@partial(jit, static_argnames=('g', 'c'), inline=True)
def k_aitchison_rbf(x, y, g, c=0):
    x = x + c
    x = x / x.sum()
    y = y + c
    y = y / y.sum()
    gm_x = gmean(x)
    gm_y = gmean(y)
    clr_x = jnp.log(x/gm_x)
    clr_y = jnp.log(y/gm_y)
    k = ((clr_x - clr_y)**2).sum()
    k *= -g
    return jnp.exp(k)


@jit
def k_chisq(x, y):
    inv_p = 1.0/x.shape[0]
    # with np.errstate(divide='ignore', invalid='ignore'):
    t1 = (x-y)**2/(x+y)
    t2 = (x-inv_p)**2/(x+inv_p)
    t3 = (y-inv_p)**2/(y+inv_p)
    return -0.5*(jnp.nansum(t1-t2-t3))


@jit
def k_hellinger(x, y):
    sqrt_inv_p = jnp.sqrt(1.0/x.shape[0])
    t1 = jnp.sqrt(x*y)-sqrt_inv_p*(jnp.sqrt(x)+jnp.sqrt(y))
    return 0.5*(1+t1.sum())


@jit
def k_js(x, y):
    inv_p = 1.0/x.shape[0]
    # with np.errstate(divide='ignore', invalid='ignore'):
    t1 = x*jnp.log((x+inv_p)/(x+y))
    t2 = y*jnp.log((y+inv_p)/(x+y))
    t3 = inv_p*jnp.log(4*inv_p**2/(x+inv_p)/(y+inv_p))
    return -0.25*jnp.nansum(t1+t2-t3)


@jit
def k_tv(x, y):
    inv_p = 1.0/x.shape[0]
    return -0.25*(abs(x-y)-abs(x-inv_p)-abs(y-inv_p)).sum()


@partial(jit, static_argnames=('b'), inline=True)
def k_hilbert1_a_inf_b_fin(x, y, b):
    inv_b = 1.0/b
    inv_p = 1.0/x.shape[0]
    t1 = 2**(inv_b)*jnp.maximum(x, y)-(x**b + y**b)**inv_b
    t2 = 2**(inv_b)*jnp.maximum(x, inv_p)-(x**b + inv_p**b)**inv_b
    t3 = 2**(inv_b)*jnp.maximum(inv_p, y)-(inv_p**b + y**b)**inv_b
    return 0.5*b*(-t1+t2+t3).sum()


@jit
def k_hilbert1_b_inf(x, y):
    inv_p = 1.0/x.shape[0]
    t1 = jnp.maximum(x, y)*(x != y)
    t2 = jnp.maximum(x, inv_p)*(x != inv_p)
    t3 = jnp.maximum(y, inv_p)*(y != inv_p)
    return 0.5*jnp.log(2)*(-t1+t2+t3).sum()


@partial(jit, static_argnames=('b'), inline=True)
def k_hilbert1_b_fin(x, y, b):
    """
    Case when a >= 1, a < inf, and b->a
    """
    # assert(b >= 1)
    p = x.shape[0]
    inv_b = 1.0/b
    xb = x**b
    yb = y**b
    xb_plus_yb = xb+yb
    inv_pb = 1.0/p**b
    # note: only f1 and t1 could encounter edge cases
    # with np.errstate(divide='ignore', invalid='ignore'):
    f1 = (xb+yb)**(inv_b-1)
    f2 = (xb+inv_pb)**(inv_b-1)
    f3 = (yb+inv_pb)**(inv_b-1)
    t1 = -(xb_plus_yb)*jnp.log(xb_plus_yb) + \
        xb*jnp.log(2*xb) + yb*jnp.log(2*yb)
    t2 = -(xb+inv_pb)*jnp.log(xb+inv_pb) + xb * \
        jnp.log(2*xb) + inv_pb*jnp.log(2*inv_pb)
    t3 = -(yb+inv_pb)*jnp.log(yb+inv_pb) + yb * \
        jnp.log(2*yb) + inv_pb*jnp.log(2*inv_pb)
    return -0.5**(1+inv_b)*jnp.nansum(f1*t1 - f2*t2 - f3*t3)


@partial(jit, static_argnames=('a', 'b'), inline=True)
def k_hilbert1_ab(x, y, a, b):
    p = x.shape[0]
    inv_a = 1.0/a
    inv_b = 1.0/b
    inv_pa = 1.0/p**a
    inv_pb = 1.0/p**b
    xa = x**a
    ya = y**a
    xb = x**b
    yb = y**b
    fac = -a*b/(a-b)*0.5**(1+inv_a+inv_b)
    t1 = (xa + ya)**inv_a - (xa+inv_pa)**inv_a - (inv_pa+ya)**inv_a
    t2 = (xb + yb)**inv_b - (xb+inv_pb)**inv_b - (inv_pb+yb)**inv_b
    return fac*jnp.nansum(2.0**inv_b*t1 - 2.0**inv_a*t2)


@partial(jit, static_argnames=('b'), inline=True)
def k_hilbert2_a_inf_b_fin(x, y, b):
    inv_p = 1.0/x.shape[0]
    inv_b = 1.0/b
    fac = -0.5/(1-2**inv_b)
    t1 = 2**inv_b*(jnp.maximum(x, y) -
                   jnp.maximum(x, inv_p) - jnp.maximum(y, inv_p))
    t2 = -(x**b+y**b)**inv_b + (x**b+inv_p**b)**inv_b + (y**b+inv_p**b)**inv_b
    return fac*jnp.sum(t1+t2)


@partial(jit, static_argnames=('a'), inline=True)
def k_hilbert2_a_fin_b_neginf(x, y, a):
    inv_p = 1.0/x.shape[0]
    inv_a = 1.0/a
    fac = -0.5/(2**inv_a-1)
    t1 = (x**a+y**a)**inv_a - (x**a+inv_p**a)**inv_a - (y**a+inv_p**a)**inv_a
    t2 = -2**inv_a*(jnp.minimum(x, y) -
                    jnp.minimum(x, inv_p) - jnp.minimum(y, inv_p))
    return fac*jnp.nansum(t1+t2)


@partial(jit, static_argnames=('a', 'b'), inline=True)
def k_hilbert2_ab(x, y, a, b):
    p = x.shape[0]
    inv_a = 1.0/a
    inv_b = 1.0/b
    xa = x**a
    ya = y**a
    # if x or y has 0s, then in t2 the first term will automatically be 0 since b is negative
    xb = x**b
    yb = y**b
    inv_pa = 1.0/p**a
    inv_pb = 1.0/p**b
    fac = -0.5/(2**inv_a - 2**inv_b)
    t1 = (xa+ya)**inv_a - (xa+inv_pa)**inv_a - (ya+inv_pa)**inv_a
    t2 = (xb+yb)**inv_b - (xb+inv_pb)**inv_b - (yb+inv_pb)**inv_b
    return fac*jnp.nansum(2.0**inv_b*t1 - 2.0**inv_a*t2)


@partial(jit, static_argnames=('t'), inline=True)
def k_hd(x, y, t):
    p = x.shape[0]
    fac = (4*jnp.pi*t)**(-(p-1)/2.0)
    t0 = jnp.sqrt(x*y).sum()
    # in case numerically bigger than 1.0, arccos will have issue
    t0 = jnp.minimum(t0, 1.0)
    t1 = jnp.exp(-1.0/t * jnp.arccos(t0)**2)
    return fac*t1


def k_hilbert1(x, y, a, b):
    # print(f'a = {a}')
    # print(f'b = {b}')
    assert(a >= 1)
    assert(b >= 0.5 and b <= a)
    if jnp.isinf(a) and a == b:
        # Note: jnp.inf == jnp.inf is True
        return k_hilbert1_b_inf(x, y)
    elif not jnp.isinf(a) and a == b:
        return k_hilbert1_b_fin(x, y, b=a)
    elif jnp.isinf(a) and b == 1:
        return 2*k_tv(x, y)
    elif jnp.isinf(a) and not jnp.isinf(b):
        # Note: with a = inf and b = 1, should be the same as 2*kmat_tv(X,Y)
        #  but kmat_tv should be faster
        return k_hilbert1_a_inf_b_fin(x, y, b=b)
    else:
        # print("diff a b case")
        return k_hilbert1_ab(x, y, a=a, b=b)


def k_hilbert2(x, y, a, b):
    assert(a >= 1)
    assert(b <= -1)
    # cannot both be inf (TODO: double check?)
    assert(not jnp.isinf(a) or not jnp.isinf(b))
    if jnp.isinf(a) and not jnp.isinf(b):
        return k_hilbert2_a_inf_b_fin(x, y, b=b)
    elif not jnp.isinf(a) and jnp.isinf(b):
        return k_hilbert2_a_fin_b_neginf(x, y, a=a)
    else:
        return k_hilbert2_ab(x, y, a=a, b=b)


# ---- kernel matrices ----


def kmat_linear(X, Y):
    return gram(k_linear, X, Y)


def kmat_js(X, Y):
    return gram(k_js, X, Y)


def kmat_chisq(X, Y):
    return gram(k_chisq, X, Y)


def kmat_tv(X, Y):
    return gram(k_tv, X, Y)


def kmat_hilbert1(X, Y, a, b):
    # print(f'a = {a}')
    # print(f'b = {b}')
    assert(a >= 1)
    assert(b >= 0.5 and b <= a)
    if jnp.isinf(a) and a == b:
        # Note: jnp.inf == jnp.inf is True
        return gram(k_hilbert1_b_inf, X, Y)
    elif not jnp.isinf(a) and a == b:
        return gram(k_hilbert1_b_fin, X, Y, b=a)
    elif jnp.isinf(a) and b == 1:
        # Note: with a = inf and b = 1, should be the same as 2*kmat_tv(X,Y)
        #  but kmat_tv should be faster
        return 2*kmat_tv(X, Y)
    elif jnp.isinf(a) and not jnp.isinf(b):
        return gram(k_hilbert1_a_inf_b_fin, X, Y, b=b)
    else:
        return gram(k_hilbert1_ab, X, Y, a=a, b=b)


def kmat_hilbert2(X, Y, a, b):
    assert(a >= 1)
    assert(b <= -1)
    # cannot both be inf (TODO: double check?)
    assert(not jnp.isinf(a) or not jnp.isinf(b))
    if jnp.isinf(a) and not jnp.isinf(b):
        return gram(k_hilbert2_a_inf_b_fin, X, Y, b=b)
    elif not jnp.isinf(a) and jnp.isinf(b):
        return gram(k_hilbert2_a_fin_b_neginf, X, Y, a=a)
    else:
        return gram(k_hilbert2_ab, X, Y, a=a, b=b)


def kmat_hd(X, Y, t):
    return gram(k_hd, X, Y, t=t)


@partial(jit, static_argnames=('g'), inline=True)
def kmat_rbf(X, Y, g):
    """
    Adapted from https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/metrics/pairwise.py#L1142
    X: jnp.array of shape (n,p)
    Y: jnp.array of shape (n,p)
    g: if None, use the median heuristic, i.e., g = 1/2*median{||x_i-x^_j||^2: i < j}
    """
    K = squared_euclidean_distances(X, Y)
    # k_triu = K[jnp.triu_indices(n=K.shape[0], k=1, m=K.shape[1])]
    # g = 1.0/jnp.median(k_triu)
    K *= -g
    return jnp.exp(K)


@partial(jit, static_argnames=('c_X', 'c_Y'), inline=True)
def kmat_aitchison(X, Y, c_X=0, c_Y=0):
    """
    X: jnp.ndarry of shape (n_sample_X, p)
    Y: jnp.ndarry of shape (n_sample_Y, p)
    c_X: a scalar or jnp.ndarray of shape (n_sample_X,)
    c_Y: a scalar or jnp.ndarray of shape (n_sample_Y,)
    """
    X = X + c_X
    Y = Y + c_Y
    gm_X = gmean(X, axis=1)
    gm_Y = gmean(Y, axis=1)
    clr_X = jnp.log(X/gm_X[:, None])
    clr_Y = jnp.log(Y/gm_Y[:, None])
    return clr_X.dot(clr_Y.T)


@partial(jit, static_argnames=('g', 'c_X', 'c_Y'), inline=True)
def kmat_aitchison_rbf(X, Y, g, c_X=0, c_Y=0):
    X = X + c_X
    Y = Y + c_Y
    gm_X = gmean(X, axis=1)
    gm_Y = gmean(Y, axis=1)
    clr_X = jnp.log(X/gm_X[:, None])
    clr_Y = jnp.log(Y/gm_Y[:, None])
    K = squared_euclidean_distances(clr_X, clr_Y)
    # k_triu = K[jnp.triu_indices(n=K.shape[0], k=1, m=K.shape[1])]
    # g = 1.0/jnp.median(k_triu)
    K *= -g
    return jnp.exp(K)


@jit
def kmat_hellinger(X, Y):
    p = X.shape[1]
    X_sqrt = jnp.sqrt(X)
    Y_sqrt = jnp.sqrt(Y)
    t1 = 0.5*squared_euclidean_distances(X_sqrt, Y_sqrt)
    t2 = 0.5 * \
        squared_euclidean_distances(X_sqrt, jnp.ones_like(Y)/jnp.sqrt(p))
    t3 = 0.5 * \
        squared_euclidean_distances(jnp.ones_like(X)/jnp.sqrt(p), Y_sqrt)
    return -0.5*t1+0.5*t2+0.5*t3
