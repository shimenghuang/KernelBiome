# %%
# libs
# ^^^^^^

from kernelbiome.helpers_analysis import (get_cfi, get_cpd, plot_cpd)
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rc
rc('font', **{'family': 'tex-gyre-termes', 'size': 6.5})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')

colors = ['tab:blue', 'tab:purple', 'tab:gray']
ccodes = mcolors.TABLEAU_COLORS


def num_df_mat(fun, h, X):
    n, p = X.shape
    res = np.zeros((n, p))
    for jj in range(p):
        ej = np.zeros(p)
        ej[jj] = 1
        res[:, jj] = (fun(X + h/2*ej) - fun(X - h/2*ej))/h
    return res

# %%
# simulate data
# ^^^^^^


rng = np.random.default_rng(2022)
p = 3
n = 200
mu = [0, 0, 0]
cov = [[1, 0.25, -0.25],
       [0.25, 1, 0.25],
       [-0.25, 0.25, 1]]

X = rng.multivariate_normal(mu, cov, n)
X = np.exp(X)
X /= X.sum(axis=1)[:, None]
print(X.shape)

# %%
# linear model (CPD v.s. PDP)
# ^^^^^^


def true_fun(X):
    return 10*X[:, 0] + 10*X[:, 1]


n_grid = 100
supp_grid = np.zeros((n_grid, p))
for jj in range(p):
    supp_grid[:, jj] = np.linspace(min(X[:, jj]), max(X[:, jj]), num=n_grid)
cpd_vals = get_cpd(X, supp_grid, true_fun, rescale=True, verbose=False)
pdp_vals = get_cpd(X, supp_grid, true_fun, rescale=False, verbose=False)

fig, axs = plt.subplots(1, 2, figsize=(
    12, 6), gridspec_kw={'width_ratios': [1, 1]})
plot_cpd(supp_grid, cpd_vals, labels=[
         f'$X^{jj+1}$' for jj in range(X.shape[1])], colors=[ccodes[c] for c in colors], axs=axs[0])
plot_cpd(supp_grid, pdp_vals, labels=[
         f'$X^{jj+1}$' for jj in range(X.shape[1])], colors=[ccodes[c] for c in colors], axs=axs[1], ylabel='PDP')
axs[0].annotate(r'$X^2$', (1, 3))
axs[0].annotate(r'$X^1$', (1, 2))
axs[0].annotate(r'$X^3$', (1, -5))
axs[1].annotate(r'$X^2$', (0.9, 6))
axs[1].annotate(r'$X^1$', (0.9, 4.5))
axs[1].annotate(r'$X^3$', (0.9, -0.5))
# axs[1].legend(loc='best', bbox_to_anchor=(1, 0.75, 0.2, 0.2))
fig.set_size_inches(3.17, 1.2)
fig.tight_layout()
fig.savefig("cpd_vs_pdp_lm.pdf.pdf", bbox_inches='tight')
plt.show()

# %%
# linear model (CFI v.s. CFI without projection v.s. PI)
# ^^^^^^

df = num_df_mat(true_fun, 1e-5, X)
cfi_vals = get_cfi(X, df, proj=True)
fi_vals = get_cfi(X, df, proj=False)

# permutation importance (PI)
B = 30
pi_vals = np.zeros(3)
baseline_mse = 0
for jj in range(3):
    for ii in range(B):
        X_loc = X.copy()
        rng.shuffle(X_loc[:, jj])
        y_loc = true_fun(X_loc)
        pi_vals[jj] += mean_squared_error(true_fun(X), y_loc) - baseline_mse
pi_vals /= B

df = pd.DataFrame({'CFI': cfi_vals,
                   'FI': fi_vals,
                   'PI': pi_vals})
df.T.round(2)

# %%
# non-linear model (CPD v.s. PDP)
# ^^^^^^


def true_fun(X):
    """
    f(x) = (1-x^2-x^3)/(1-x^3)
    """
    return 10*(1-X[:, 1]-X[:, 2])/(1-X[:, 2])


n_grid = 100
supp_grid = np.zeros((n_grid, p))
for jj in range(p):
    supp_grid[:, jj] = np.linspace(min(X[:, jj]), max(X[:, jj]), num=n_grid)
cpd_vals = get_cpd(X, supp_grid, true_fun, rescale=True, verbose=False)
pdp_vals = get_cpd(X, supp_grid, true_fun, rescale=False, verbose=False)

fig, axs = plt.subplots(1, 2, figsize=(
    12, 6), gridspec_kw={'width_ratios': [1, 1]})
plot_cpd(supp_grid, cpd_vals, labels=[
         f'$X^{jj+1}$' for jj in range(X.shape[1])], colors=[ccodes[c] for c in colors], axs=axs[0])
plot_cpd(supp_grid, pdp_vals, labels=[
         f'$X^{jj+1}$' for jj in range(X.shape[1])], colors=[ccodes[c] for c in colors], axs=axs[1], ylabel='PDP')
axs[0].annotate(r'$X^1$', (0.9, 3.5))
axs[0].annotate(r'$X^3$', (0.9, 0.5))
axs[0].annotate(r'$X^2$', (0.9, -4))
axs[1].annotate(r'$X^1$', (0.9, 0.5))
axs[1].annotate(r'$X^2$', (0.9, -10))
axs[1].annotate(r'$X^3$', (0.9, -30))
# axs[1].legend(loc='best', bbox_to_anchor=(1, 0.75, 0.2, 0.2))
fig.set_size_inches(3.17, 1.2)
fig.tight_layout()
fig.savefig("cpd_vs_pdp_nlm.pdf.pdf", bbox_inches='tight')
plt.show()

# %%
# non-linear model (CFI v.s. CFI without projection v.s. PI)
# ^^^^^^

df = num_df_mat(true_fun, 1e-5, X)
cfi_vals = get_cfi(X, df, proj=True)
cfi_no_proj_vals = get_cfi(X, df, proj=False)

# permutation importance (PI)
B = 30
pi_vals = np.zeros(3)
baseline_mse = 0
for jj in range(3):
    for ii in range(B):
        X_loc = X.copy()
        rng.shuffle(X_loc[:, jj])
        y_loc = true_fun(X_loc)
        pi_vals[jj] += mean_squared_error(true_fun(X), y_loc) - baseline_mse
pi_vals /= B

# %%
