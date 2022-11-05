# %%
# libs
# ^^^^^^

from os.path import join
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rc, rcParams

width = 3.17
fontsize = 6
# rc('font', **{'family': 'tex-gyre-termes', 'size': 6.5})
# rc('font', **{'family': 'tex-gyre-termes', 'size': 3.8})
rc('font', **{'size': fontsize})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')

# %%
# paths
# ^^^^^^

print(f'cwd: {os.getcwd()}')
if os.getcwd().endswith('consistency'):
    res_path = "results"
else:
    # assuming running from kernelbiome_tmp
    res_path = "experiments/consistency/results"

# %%
# load results
# ^^^^^^

cfi_mse_df = pd.read_csv(join(res_path, "cfi_mse_seed_0.csv"))
for ii in range(1, 100):
    try:
        tmp = pd.read_csv(join(res_path, f"cfi_mse_seed_{ii}.csv"))
        cfi_mse_df = pd.concat([cfi_mse_df, tmp])
    except:
        print(f"seed {ii} not found.")
        pass
print(cfi_mse_df.shape)

cpd_mse_df = pd.read_csv(join(res_path, "cpd_mse_seed_0.csv"))
for ii in range(1, 100):
    try:
        tmp = pd.read_csv(join(res_path, f"cpd_mse_seed_{ii}.csv"))
        cpd_mse_df = pd.concat([cpd_mse_df, tmp])
    except:
        print(f"seed {ii} not found.")
        pass
print(cpd_mse_df.shape)

# %%
# plot
# ^^^^^^

colors = ['tab:blue', 'tab:orange']
ccodes = mcolors.TABLEAU_COLORS

ns = cfi_mse_df.columns.to_numpy()

fig, ax1 = plt.subplots()
res1 = ax1.boxplot(
    cfi_mse_df,
    positions=np.arange(1, len(ns)+1)-0.15,
    notch=True,
    widths=0.2,
    boxprops=dict(
        facecolor=ccodes[colors[0]],
        edgecolor='white'),
    flierprops=dict(markersize=2),
    patch_artist=True)
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(res1[element], color='k')

for patch in res1['boxes']:
    patch.set_facecolor('tab:blue')
ax1.set_xlabel('n')
ax1.set_ylabel('MSD (CFI)', color='tab:blue')

ax2 = ax1.twinx()
res2 = ax2.boxplot(
    cpd_mse_df,
    positions=np.arange(1, len(ns)+1)+0.15,
    notch=True,
    widths=0.2,
    boxprops=dict(
        facecolor=ccodes[colors[1]],
        edgecolor='white'),
    flierprops=dict(markersize=2),
    patch_artist=True)
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(res2[element], color='k')

for patch in res2['boxes']:
    patch.set_facecolor('tab:orange')
ax2.set_ylabel('MSD (CPD)', color='tab:orange')

ax1.set_xticks(range(1, len(ns)+1))
ax1.set_xticklabels(ns)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

for spine in ax1.spines.values():
    spine.set(color='gray', linewidth=1, alpha=0.2)
for spine in ax2.spines.values():
    spine.set_visible(False)

ax1.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=True)
ax2.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=True)

# plt.style.use("seaborn-white")
fig.set_size_inches(2.6, 1.6)
fig.savefig("consistency_cfi_and_cpd.pdf", bbox_inches='tight')
plt.show()

# %%
