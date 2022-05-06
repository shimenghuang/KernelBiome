from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from src.helpers_jax import *


@jit
def d2_linear(x, y):
    """
    Squared 2-norm.
    """
    return jnp.sum((x-y)**2)


@jit
def d2_rbf(x, y, g):
    """
    Squared RBF metric.
    """
    return 2-2*jnp.exp(-g*jnp.sum(((x-y)**2)))


@jit
def d2_aitchison(x, y, c=0):
    """
    Squared Aitchison metric.
    """
    x = x + c
    x = x / x.sum()
    y = y + c
    y = y / y.sum()
    p = x.shape[0]
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    return jnp.sum(0.5/p*((jnp.log(x.T/x) - jnp.log(y.T/y))**2))


@jit
def d2_aitchison_rbf(x, y, g, c=0):
    """
    Squared Aitchison-RBF metric.
    """
    x = x + c
    x = x / x.sum()
    y = y + c
    y = y / y.sum()
    gm_x = gmean(x)
    gm_y = gmean(y)
    clr_x = jnp.log(x/gm_x)
    clr_y = jnp.log(y/gm_y)
    return 2-2*jnp.exp(-g*jnp.sum(((clr_x-clr_y)**2)))


@jit
def d2_chisq(x, y):
    """
    Squared Chi-square metric.
    """
    t = (x-y)**2/(x+y)
    return jnp.nansum(t)


@jit
def d2_hellinger(x, y):
    """
    Squared Hellinger metric.
    """
    t1 = jnp.sqrt(x)-jnp.sqrt(y)
    return 0.5*jnp.sum(((t1)**2))


@jit
def d2_js(x, y):
    """
    Squared Jenson-Shannon metric.
    """
    t1 = x*jnp.log(2*x/(x+y))
    t2 = y*jnp.log(2*y/(x+y))
    return jnp.maximum(0.5*jnp.nansum(t1+t2), 0.0)


@jit
def d2_tv(x, y):
    """
    Squared total variation metric.
    """
    return 0.5*jnp.sum(abs(x-y))


@jit
def d2_hilbert1_a_inf_b_fin(x, y, b):
    """
    Squared Hilbertian metric by Topsoe (a >= 1, b \in [0.5, a]) when a is infinite but b is finite.
    """
    inv_b = 1.0/b
    t1 = 2**(inv_b)*jnp.maximum(x, y)
    t2 = (x**b + y**b)**inv_b
    return b*jnp.sum((t1-t2))


@jit
def d2_hilbert1_b_inf(x, y):
    """
    Squared Hilbertian metric by Topsoe (a >= 1, b \in [0.5, a]) when a and b are equal and both go to infinity.
    """
    fac = jnp.maximum(x, y)
    # t1 = jnp.log(2)*(x > y)
    # t2 = jnp.log(2)*(y > x)
    t = jnp.log(2)*(x != y)
    return jnp.sum(fac*t)


@jit
def d2_hilbert1_b_fin(x, y, b):
    """
    Squared Hilbertian metric by Topsoe (a >= 1, b \in [0.5, a]) when a and b are equal and finite.
    """
    xb = x**b
    yb = y**b
    xb_plus_yb = xb+yb
    fac = (xb_plus_yb/2.0)**(1.0/b)
    t1 = xb/xb_plus_yb*jnp.log(2.0*xb/xb_plus_yb)
    t2 = yb/xb_plus_yb*jnp.log(2.0*yb/xb_plus_yb)
    return jnp.nansum(fac*(t1+t2))


@jit
def d2_hilbert1_ab(x, y, a, b):
    """
    Squared Hilbertian metric by Topsoe (a >= 1, b \in [0.5, a]) when a and b are not equal but both finite.
    """
    inv_a = 1.0/a
    inv_b = 1.0/b
    fac = a*b/(a-b)*0.5**(inv_a+inv_b)
    t1 = 2**inv_b*(x**a + y**a)**inv_a
    t2 = 2**inv_a*(x**b + y**b)**inv_b
    return jnp.maximum(fac*(jnp.sum(t1-t2)), 0.0)


@jit
def d2_hilbert2_a_inf_b_fin(x, y, b):
    """
    Squared Hilbertian metric by Hein and Bousquet (a >= 1, b <= -1) when a is infinite but b is finite.
    """
    inv_b = 1.0/b
    t1 = 2**inv_b*jnp.maximum(x, y) - (x**b+y**b)**inv_b
    t2 = 1-2**inv_b
    return jnp.sum(t1)/t2


@jit
def d2_hilbert2_a_fin_b_neginf(x, y, a):
    """
    Squared Hilbertian metric by Hein and Bousquet (a >= 1, b <= -1) when a is finite but b is infinite.
    """
    inv_a = 1.0/a
    t1 = (x**a+y**a)**inv_a - 2**inv_a*jnp.minimum(x, y)
    t2 = 2**inv_a-1
    return jnp.sum(t1)/t2


@jit
def d2_hilbert2_ab(x, y, a, b, tol=1e-7):
    """
    Squared Hilbertian metric Hein and Bousquet (a >= 1, b <= -1) when a and b are both finite.
    """
    inv_a = 1.0/a
    inv_b = 1.0/b
    two_to_inv_a = 2**inv_a
    two_to_inv_b = 2**inv_b
    # note: could encounter the case 0^{-1} e.g. when x == y == 0 and b = -1
    num = two_to_inv_b*(x**a + y**a)**inv_a - two_to_inv_a*(x**b + y**b)**inv_b
    den = two_to_inv_a-two_to_inv_b
    return jnp.maximum(jnp.nansum(num/den), 0.0)


@jit
def d2_hd(x, y, t):
    """
    Squared heat diffusion metric.
    """
    p = x.shape[0]
    fac = 2*(2*jnp.pi*t)**(-p/2.0)
    t0 = jnp.sum(jnp.sqrt(x*y))
    # in case numerically bigger than 1.0, arccos will have issue
    t0 = jnp.minimum(t0, 1.0)
    t1 = jnp.exp(-1.0/t * jnp.arccos(t0)**2)
    return fac*(1-t1)


# def d_unifrac(x, y, n_total_branch):
#     """
#     TODO: check the original paper https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=UniFrac%3A+a+New+Phylogenetic+Method+for+Comparing+Microbial+Communities&btnG=
#     Here the implementation follows the description on wikipedia.
#     """
#     x_split = x.split(';')
#     y_split = y.split(';')
#     n_unshared_branch = 0.0
#     for ii in range(min(len(x_split), len(y_split))):
#         if x_split[ii] != y_split[ii]:
#             n_unshared_branch += 1
#     n_unshared_branch += abs(len(x_split)-len(y_split))
#     return n_unshared_branch/n_total_branch


# def d2_pp(x, y, q, tol=1e-7):
#     assert(q > 0)
#     res = sum(x**(2*q)) + sum(y**(2*q)) - 2*(x**q).dot(y**q)
#     if res < 0 and np.abs(res) < tol:
#         res = 0.0
#     return res
