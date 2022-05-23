from itertools import product
import numpy as np
from .helpers_jax import wrap
from .kernels_jax import *
from .kernels_weighted_jax import *


def get_rbf_bandwidth(m):
    """
    m: value of median of squared euc. dist.
    """
    return [np.sqrt(m), 0.5*m, m, m**1.5, m**2, m**2.5, 10*m, 100*m]


def default_kernel_params_grid(g1=None, g2=None):
    if g1 is None:
        grid_rbf = np.logspace(-2, 2, 5)
    else:
        grid_rbf = get_rbf_bandwidth(g1)
    if g2 is None:
        grid_aitrbf = np.logspace(-2, 2, 5)
    else:
        grid_aitrbf = get_rbf_bandwidth(g2)
    kernel_params_dict = {
        'linear': None,
        'rbf': {'g': grid_rbf},
        # Note: hilbert1, only valid when a >= 1, 0.5 <= b <= a
        'generalized-js': {'a': [1, 10, np.inf], 'b': [0.5, 1, 10, np.inf]},
        # Note: hilbert2, only valid when a >= 1 b <= -1 and not both a, b are inf
        'hilbertian': {'a': [1, 10, np.inf], 'b': [-1, -10, -np.inf]},
        'aitchison': {'c': np.logspace(-7, -3, 5)},
        'aitchison-rbf': {'c': np.logspace(-7, -3, 5), 'g': grid_aitrbf},
        'heat-diffusion': {'t': np.linspace(0.9, 1.1, 5)*0.25/np.pi}
    }
    return kernel_params_dict


def default_weighted_kernel_params_grid(w_unifrac, g1=None, g2=None):
    if g1 is None:
        grid_rbf = np.logspace(-2, 2, 5)
    else:
        grid_rbf = get_rbf_bandwidth(g1)
    if g2 is None:
        grid_aitrbf = np.logspace(-2, 2, 5)
    else:
        grid_aitrbf = get_rbf_bandwidth(g2)
    kernel_params_dict = {
        'linear_weighted': {'w': w_unifrac},
        'rbf_weighted': {'g': grid_rbf, "w": w_unifrac},
        # Note: only valid
        'generalized-js_weighted': {'a': [1, 10, np.inf], 'b': [0.5, 1, 10, np.inf], "w": w_unifrac},
        # Note: only valid when a >= 1 b <= -1 and not both a, b are inf
        'hilbertian_weighted': {'a': [1, 10, np.inf], 'b': [-1, -10, -np.inf], "w": w_unifrac},
        'aitchison_weighted': {'c': np.logspace(-7, -3, 5), 'w': w_unifrac},
        'aitchison-rbf_weighted': {'c': np.logspace(-7, -3, 5), 'g': grid_aitrbf, 'w': w_unifrac},
        'heat-diffusion_weighted': {'t': np.linspace(0.9, 1.1, 5)*0.25/np.pi}
    }
    return kernel_params_dict


def get_kmat_with_params(kernel_params_dict):
    kmat_with_params = {}
    for kname, params in kernel_params_dict.items():
        if kname == 'linear':
            kmat_with_params[kname] = kmat_linear
        if kname == 'rbf':
            for g in params['g']:
                kmat_with_params[f'{kname}_g_{g}'] = wrap(kmat_rbf, g=g)
        if kname == 'generalized-js':
            params_ab = list(product(params['a'], params['b']))
            for ab in params_ab:
                if ab[0] >= 1 and ab[1] >= 0.5 and ab[1] <= ab[0]:
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(
                        kmat_hilbert1, a=ab[0], b=ab[1])
        if kname == 'hilbertian':
            params_ab = list(product(params['a'], params['b']))
            for ab in params_ab:
                if ab[0] >= 1 and ab[1] <= -1 and (not jnp.isinf(ab[0]) or not jnp.isinf(ab[1])):
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(
                        kmat_hilbert2, a=ab[0], b=ab[1])
        if kname == 'aitchison':
            for c in params['c']:
                kmat_with_params[f'{kname}_c_{c}'] = wrap(
                    kmat_aitchison, c_X=c, c_Y=c)
        if kname == 'aitchison-rbf':
            params_cg = list(product(params['c'], params['g']))
            for cg in params_cg:
                kmat_with_params[f'{kname}_c_{cg[0]}_g_{cg[1]}'] = wrap(
                    kmat_aitchison_rbf, c_X=cg[0], c_Y=cg[0], g=cg[1])
        if kname == 'heat-diffusion':
            for t in params['t']:
                kmat_with_params[f'{kname}_t_{t}'] = wrap(kmat_hd, t=t)
    return kmat_with_params


def get_weighted_kmat_with_params(kernel_params_dict, w_unifrac=None):
    kmat_with_params = {}
    for kname, params in kernel_params_dict.items():
        if kname == 'linear_weighted':
            kmat_with_params[f'{kname}'] = wrap(
                kmat_linear_weighted, w=w_unifrac)
        if kname == 'rbf_weighted':
            for g in params['g']:
                kmat_with_params[f'{kname}_g_{g}'] = wrap(
                    kmat_rbf_weighted, g=g, w=w_unifrac)
        if kname == 'generalized-js_weighted':
            params_ab = list(product(params['a'], params['b']))
            for ab in params_ab:
                if ab[0] >= 1 and ab[1] >= 0.5 and ab[1] <= ab[0]:
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(kmat_hilbert1_weighted, a=ab[0], b=ab[1],
                                                                            w=w_unifrac)
        if kname == 'hilbertian_weighted':
            params_ab = list(product(params['a'], params['b']))
            for ab in params_ab:
                if ab[0] >= 1 and ab[1] <= -1 and (not jnp.isinf(ab[0]) or not jnp.isinf(ab[1])):
                    kmat_with_params[f'{kname}_a_{ab[0]}_b_{ab[1]}'] = wrap(kmat_hilbert2_weighted, a=ab[0], b=ab[1],
                                                                            w=w_unifrac)
        if kname == 'aitchison_weighted':
            for c in params['c']:
                kmat_with_params[f'{kname}_c_{c}'] = wrap(
                    kmat_aitchison_weighted, c=c, w=w_unifrac)
        if kname == 'aitchison-rbf_weighted':
            params_cg = list(product(params['c'], params['g']))
            for cg in params_cg:
                kmat_with_params[f'{kname}_c_{cg[0]}_g_{cg[1]}'] = wrap(
                    kmat_aitchison_rbf_weighted, g=cg[1], c=cg[0], w=w_unifrac)
        if kname == 'heat-diffusion_weighted':
            for t in params['t']:
                kmat_with_params[f'{kname}_t_{t}'] = wrap(
                    kmat_hd_weighted, t=t, w=w_unifrac)

    return kmat_with_params
