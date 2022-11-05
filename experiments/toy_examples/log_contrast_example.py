# %%
# libs
# ^^^^^^

# Note: add path so it also works for interactive run in vscode...
import sys  # nopep8
sys.path.insert(0, "../../")  # nopep8
import numpy as np
from kernelbiome.helpers_analysis import (get_cfi, get_cpd, plot_cpd, plot_cfi)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rc
rc('font', **{'size': 6})  # 'family': 'tex-gyre-termes',
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')

# %%
# helper functions
# ^^^^^^


def true_fun(X):
    beta = np.array([2, -1, -1, 0])
    return np.log(X).dot(beta)


def d_true_fun(X):
    beta = np.array([2, -1, -1, 0])
    return 1/X * beta

# %%
# simulate data
# ^^^^^^


rng = np.random.default_rng(42)
n = 50
p = 4
X = np.exp(rng.normal(0, 1, (n, p)))
X /= X.sum(axis=1)[:, None]

# %%
# make plots
# ^^^^^^

# print(list(mcolors.TABLEAU_COLORS.keys()))
# 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
# 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
ccodes = mcolors.TABLEAU_COLORS
# cnames = list(mcolors.TABLEAU_COLORS.keys())
# ccodes = mcolors.TABLEAU_COLORS
colors = ['tab:orange', 'tab:green', 'tab:blue', 'tab:gray']
colors_all = [ccodes[c] for c in colors]

cfi_vals = get_cfi(X, d_true_fun(X))
cpd_vals = get_cpd(X, X, true_fun)

plt.style.use("seaborn-white")
fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
plot_cpd(X, cpd_vals, labels=[
         f'$X^{jj+1}$' for jj in range(X.shape[1])], colors=colors_all, axs=axs[0])
axs[0].annotate(r'$X^1$', (0.7, 4))
axs[0].annotate(r'$X^4$', (0.82, 0))
axs[0].annotate(r'$X^3$', (0.82, -3))
axs[0].annotate(r'$X^2$', (0.7, -3.5))
plot_cfi(cfi_vals, fmt='%.2f', labels=[f'$X^{jj+1}$' for jj in range(
    X.shape[1])], colors=colors_all, axs=axs[1], ascending=False)
fig.set_size_inches(3.17, 1.2)
fig.savefig("log_contrast_example.pdf", bbox_inches='tight')

# %%
