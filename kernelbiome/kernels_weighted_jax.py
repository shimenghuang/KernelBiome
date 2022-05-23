from functools import partial
import jax.numpy as jnp
from jax import jit
from .helpers_jax import gmean, gram

# ---- kernel functions on vectors ----


@jit
def k_linear_weighted(x, y, w):
    inv_p = 1.0 / x.shape[0]
    mat = w * (x[..., jnp.newaxis]@y[..., jnp.newaxis].T - inv_p * (x[..., jnp.newaxis] + y[..., jnp.newaxis]) +
               inv_p**2)
    return mat.sum()


@partial(jit, static_argnames=('g'), inline=True)
def k_rbf_weighted(x, y, g, w):
    mesh = jnp.stack(jnp.meshgrid(x, y))
    x1 = mesh[0].reshape(-1)
    x2 = mesh[1].reshape(-1)
    w = w.reshape(-1)
    vec = - g * jnp.sum((w * (x1 - x2) ** 2))
    return jnp.exp(vec)


@partial(jit, static_argnames=('t'), inline=True)
def k_hd_weighted(x, y, t, w):
    p = x.shape[0]
    fac = (4*jnp.pi*t)**(-(p-1)/2.0)
    t0 = jnp.sqrt(x).dot(w).dot(jnp.sqrt(y))
    # in case numerically bigger than 1.0, arccos will have issue
    t0 = jnp.minimum(t0, 1.0)
    t1 = jnp.exp(-1.0/t * jnp.arccos(t0)**2)
    return fac*t1


@partial(jit, static_argnames=('c'), inline=True)
def k_aitchison_weighted(x, y, c, w):
    x = x + c
    x = x / x.sum()
    y = y + c
    y = y / y.sum()
    gm_x = gmean(x)
    gm_y = gmean(y)

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    clr_x = jnp.log(x / gm_x)
    clr_y = jnp.log(y / gm_y)

    return jnp.sum(w * (clr_x * clr_y))


@partial(jit, static_argnames=('g', 'c'), inline=True)
def k_aitchison_rbf_weighted(x, y, g, c, w):
    x = x + c
    x = x / x.sum()
    y = y + c
    y = y / y.sum()
    gm_x = gmean(x)
    gm_y = gmean(y)

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    clr_x = jnp.log(x / gm_x)
    clr_y = jnp.log(y / gm_y)
    k = jnp.sum(w * ((clr_x - clr_y) ** 2))
    k *= -g

    return jnp.exp(k)


@jit
def k_hilbert1_b_inf_weighted(x, y, w):
    inv_p = 1.0 / x.shape[0]

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    t1 = jnp.maximum(x, y) * (x != y)
    t2 = jnp.maximum(x, inv_p) * (x != inv_p)
    t3 = jnp.maximum(y, inv_p) * (y != inv_p)

    return 0.5 * jnp.log(2) * jnp.sum(w * (-t1 + t2 + t3))


@partial(jit, static_argnames=('b'), inline=True)
def k_hilbert1_b_fin_weighted(x, y, b, w):
    """
    Case when a >= 1, a < inf, and b->a
    """
    p = x.shape[0]

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    inv_b = 1.0 / b
    xb = x ** b
    yb = y ** b
    xb_plus_yb = xb + yb
    inv_pb = 1.0 / p ** b
    # note: only f1 and t1 could encounter edge cases
    # with np.errstate(divide='ignore', invalid='ignore'):
    f1 = (xb + yb) ** (inv_b - 1)
    f2 = (xb + inv_pb) ** (inv_b - 1)
    f3 = (yb + inv_pb) ** (inv_b - 1)
    t1 = -(xb_plus_yb) * jnp.log(xb_plus_yb) + \
        xb * jnp.log(2 * xb) + yb * jnp.log(2 * yb)
    t2 = -(xb + inv_pb) * jnp.log(xb + inv_pb) + xb * \
        jnp.log(2 * xb) + inv_pb * jnp.log(2 * inv_pb)
    t3 = -(yb + inv_pb) * jnp.log(yb + inv_pb) + yb * \
        jnp.log(2 * yb) + inv_pb * jnp.log(2 * inv_pb)
    return -0.5 ** (1 + inv_b) * jnp.nansum(w * (f1 * t1 - f2 * t2 - f3 * t3))


@partial(jit, static_argnames=('b'), inline=True)
def k_hilbert1_a_inf_b_fin_weighted(x, y, b, w):
    inv_b = 1.0 / b
    inv_p = 1.0 / x.shape[0]

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    t1 = 2 ** (inv_b) * jnp.maximum(x, y) - (x ** b + y ** b) ** inv_b
    t2 = 2 ** (inv_b) * jnp.maximum(x, inv_p) - (x ** b + inv_p ** b) ** inv_b
    t3 = 2 ** (inv_b) * jnp.maximum(inv_p, y) - (inv_p ** b + y ** b) ** inv_b
    return 0.5 * b * jnp.sum(w * (-t1 + t2 + t3))


@partial(jit, static_argnames=('a', 'b'), inline=True)
def k_hilbert1_ab_weighted(x, y, a, b, w):
    p = x.shape[0]

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    inv_a = 1.0 / a
    inv_b = 1.0 / b
    inv_pa = 1.0 / p ** a
    inv_pb = 1.0 / p ** b
    xa = x ** a
    ya = y ** a
    xb = x ** b
    yb = y ** b
    fac = -a * b / (a - b) * 0.5 ** (1 + inv_a + inv_b)
    t1 = (xa + ya) ** inv_a - (xa + inv_pa) ** inv_a - (inv_pa + ya) ** inv_a
    t2 = (xb + yb) ** inv_b - (xb + inv_pb) ** inv_b - (inv_pb + yb) ** inv_b
    return fac * jnp.sum(w * (2.0 ** inv_b * t1 - 2.0 ** inv_a * t2))


def k_hilbert1_weighted(x, y, a, b, w):
    assert(a >= 1)
    assert(b >= 0.5 and b <= a)
    if jnp.isinf(a) and a == b:
        # Note: jnp.inf == jnp.inf is True
        return k_hilbert1_b_inf_weighted(x, y, w)
    elif not jnp.isinf(a) and a == b:
        return k_hilbert1_b_fin_weighted(x, y, b=a, w=w)
    elif jnp.isinf(a) and not jnp.isinf(b):
        # Note: with a = inf and b = 1, should be the same as 2*kmat_tv(X,Y)
        #  but kmat_tv should be faster
        return k_hilbert1_a_inf_b_fin_weighted(x, y, b=b, w=w)
    else:
        return k_hilbert1_ab_weighted(x, y, a=a, b=b, w=w)


@partial(jit, static_argnames=('b'), inline=True)
def k_hilbert2_a_inf_b_fin_weighted(x, y, b, w):
    inv_p = 1.0 / x.shape[0]

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    inv_b = 1.0 / b
    fac = -0.5 / (1 - 2 ** inv_b)
    t1 = 2 ** inv_b * (jnp.maximum(x, y) -
                       jnp.maximum(x, inv_p) - jnp.maximum(y, inv_p))
    t2 = -(x ** b + y ** b) ** inv_b + (x ** b + inv_p **
                                        b) ** inv_b + (y ** b + inv_p ** b) ** inv_b
    return fac * jnp.sum(w * (t1 + t2))


@partial(jit, static_argnames=('a'), inline=True)
def k_hilbert2_a_fin_b_neginf_weighted(x, y, a, w):
    inv_p = 1.0 / x.shape[0]

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    inv_a = 1.0 / a
    fac = -0.5 / (2 ** inv_a - 1)
    t1 = (x ** a + y ** a) ** inv_a - (x ** a + inv_p **
                                       a) ** inv_a - (y ** a + inv_p ** a) ** inv_a
    t2 = -2 ** inv_a * (jnp.minimum(x, y) -
                        jnp.minimum(x, inv_p) - jnp.minimum(y, inv_p))
    return fac * jnp.nansum(w * (t1 + t2))


@partial(jit, static_argnames=('a', 'b'), inline=True)
def k_hilbert2_ab_weighted(x, y, a, b, w):
    p = x.shape[0]

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    inv_a = 1.0 / a
    inv_b = 1.0 / b
    xa = x ** a
    ya = y ** a
    # if x or y has 0s, then in t2 the first term will automatically be 0 since b is negative
    xb = x ** b
    yb = y ** b
    inv_pa = 1.0 / p ** a
    inv_pb = 1.0 / p ** b
    fac = -0.5 / (2 ** inv_a - 2 ** inv_b)
    t1 = (xa + ya) ** inv_a - (xa + inv_pa) ** inv_a - (ya + inv_pa) ** inv_a
    t2 = (xb + yb) ** inv_b - (xb + inv_pb) ** inv_b - (yb + inv_pb) ** inv_b
    return fac * jnp.nansum(w * (2.0 ** inv_b * t1 - 2.0 ** inv_a * t2))


def k_hilbert2_weighted(x, y, a, b, w):
    assert(a >= 1)
    assert(b <= -1)
    assert(not jnp.isinf(a) or not jnp.isinf(b))
    if jnp.isinf(a) and not jnp.isinf(b):
        return k_hilbert2_a_inf_b_fin_weighted(x, y, b=b, w=w)
    elif not jnp.isinf(a) and jnp.isinf(b):
        return k_hilbert2_a_fin_b_neginf_weighted(x, y, a=a, w=w)
    else:
        return k_hilbert2_ab_weighted(x, y, a=a, b=b, w=w)


@jit
def k_hellinger_weighted(x, y, w):
    p = x.shape[0]
    sqrt_inv_p = jnp.sqrt(1.0 / p)

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    t1 = jnp.sqrt(x * y) - sqrt_inv_p * (jnp.sqrt(x) + jnp.sqrt(y))
    # return 0.5*(1+t1.sum())
    return 0.5 * jnp.sum(w * (1 / p + t1))


@jit
def k_chisq_weighted(x, y, w):
    inv_p = 1.0 / x.shape[0]

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    # with np.errstate(divide='ignore', invalid='ignore'):
    t1 = (x - y) ** 2 / (x + y)
    t2 = (x - inv_p) ** 2 / (x + inv_p)
    t3 = (y - inv_p) ** 2 / (y + inv_p)
    return -0.5 * (jnp.nansum(w * (t1 - t2 - t3)))


@jit
def k_js_weighted(x, y, w):
    p = x.shape[0]
    inv_p = 1.0 / p

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    # with np.errstate(divide='ignore', invalid='ignore'):
    t1 = x * jnp.log((x + inv_p) / (x + y))
    t2 = y * jnp.log((y + inv_p) / (x + y))
    t3 = inv_p * jnp.log(4 * inv_p ** 2 / (x + inv_p) / (y + inv_p))
    return -0.25 * jnp.nansum(w * (t1 + t2 - t3))


@jit
def k_tv_weighted(x, y, w):
    p = x.shape[0]
    inv_p = 1.0 / p

    mesh = jnp.stack(jnp.meshgrid(x, y))
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    w = w.reshape(-1)

    return -0.25 * jnp.sum(w * (abs(x - y) - abs(x - inv_p) - abs(y - inv_p)))


# ---- kernel matrices ----


def kmat_linear_weighted(X, Y, w):
    return gram(k_linear_weighted, X, Y, w=w)


def kmat_rbf_weighted(X, Y, g, w):
    return gram(k_rbf_weighted, X, Y, g=g, w=w)


def kmat_hd_weighted(X, Y, t, w):
    return gram(k_hd_weighted, X, Y, t=t, w=w)


def kmat_aitchison_weighted(x, y, c, w):
    return gram(k_aitchison_weighted, x, y, c=c, w=w)


def kmat_aitchison_rbf_weighted(x, y, g, c, w):
    return gram(k_aitchison_rbf_weighted, x, y, g=g, c=c, w=w)


def kmat_hilbert1_weighted(X, Y, a, b, w):
    # print(f'a = {a}')
    # print(f'b = {b}')
    assert(a >= 1)
    assert(b >= 0.5 and b <= a)
    if jnp.isinf(a) and a == b:
        # Note: jnp.inf == jnp.inf is True
        return gram(k_hilbert1_b_inf_weighted, X, Y, w=w)
    elif not jnp.isinf(a) and a == b:
        return gram(k_hilbert1_b_fin_weighted, X, Y, b=a, w=w)
    # TODO : check the case of a=infty and b=1 is still caught in the clause below
    # elif jnp.isinf(a) and b == 1:
    #    return 2*kmat_tv_weighted(X, Y, w=w)
    elif jnp.isinf(a) and not jnp.isinf(b):
        # Note: with a = inf and b = 1, should be the same as 2*kmat_tv(X,Y)
        #  but kmat_tv should be faster
        return gram(k_hilbert1_a_inf_b_fin_weighted, X, Y, b=b, w=w)
    else:
        # print("diff a b case")
        return gram(k_hilbert1_ab_weighted, X, Y, a=a, b=b, w=w)


def kmat_hilbert2_weighted(X, Y, a, b, w):
    assert(a >= 1)
    assert(b <= -1)
    # cannot both be inf (TODO: double check?)
    assert(not jnp.isinf(a) or not jnp.isinf(b))
    if jnp.isinf(a) and not jnp.isinf(b):
        return gram(k_hilbert2_a_inf_b_fin_weighted, X, Y, b=b, w=w)
    elif not jnp.isinf(a) and jnp.isinf(b):
        return gram(k_hilbert2_a_fin_b_neginf_weighted, X, Y, a=a, w=w)
    else:
        return gram(k_hilbert2_ab_weighted, X, Y, a=a, b=b, w=w)


def kmat_chisq_weighted(x, y, w):
    return gram(k_chisq_weighted, x, y, w=w)


def kmat_hellinger_weighted(x, y, w):
    return gram(k_hellinger_weighted, x, y, w=w)


def kmat_js_weighted(x, y, w):
    return gram(k_js_weighted, x, y, w=w)


def kmat_tv_weighted(x, y, w):
    return gram(k_tv_weighted, x, y, w=w)
