# %%
# load libs
# ^^^^^^

import sys  # nopep8
sys.path.insert(0, "../")  # nopep8

from matplotlib import rc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd
from src.cfi import *
import load_cirrhotic

seed_num = 2022

rc('font', **{'family': 'tex-gyre-termes', 'size': 6.5})
rc('text', usetex=True)
rc('text.latex',
   preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')

# plt.cm.tab10(range(10)), plt.cm.tab20(range(20)), plt.cm.tab20b(range(20)),
colors_all = np.vstack([plt.cm.tab20c(range(20)), plt.cm.tab20b(range(20))])
colors_all = np.unique(colors_all, axis=0)
print(colors_all.shape)

# %%
# load resutls
# ^^^^^^^

data_path = "/Users/hrt620/Documents/projects/kernelbiome_proj/kernelbiome_clean/data/MLRepo/qin2014"
X_df, y, label = load_cirrhotic.main(data_path=data_path, seed_num=seed_num)

# Unweighted CFI
cirrhotic_cfi_vals = np.load("output/cirrhotic_unweighted_cfi_vals.npy")

# Weighted CFI (M^A)
cirrhotic_ma_cfi_vals = np.load("output/cirrhotic_weighted_MA_cfi_vals.npy")

# %%
# Create a 3-subplot plot for the appendix
# ^^^^^^

selected_comp = np.argsort(-np.abs(cirrhotic_cfi_vals))

short_label = [lbl.split(';')[-1] for lbl in label]
short_label = [lbl.split('__')[1] for lbl in short_label]
short_label = np.array([lbl.replace('_', ' ') for lbl in short_label])

X_all = X_df.to_numpy().astype('float').T
X_all /= X_all.sum(axis=1)[:, None]

# Gini impurity feature importance
rf = RandomForestClassifier(
    max_depth=np.sqrt(X_all.shape[0]), random_state=seed_num)
rf.fit(X_all, y)
cirrhotic_gi_vals = rf.feature_importances_

# plot CFI, weighted CFI, and GI
cfi_and_gi_df = pd.DataFrame({
    'label': short_label[selected_comp],
    'cfi_vals': cirrhotic_cfi_vals[selected_comp],
    'ma_cfi_vals': cirrhotic_ma_cfi_vals[selected_comp],
    'pi_vals': cirrhotic_gi_vals[selected_comp]
})

fig, axs = plt.subplots(1, 3,
                        gridspec_kw={'width_ratios': [2, 2, 1]},
                        constrained_layout=True)

cfi_and_gi_df.plot(
    x="label", y="cfi_vals",
    kind="barh", ax=axs[0],
    color=tuple(colors_all[0]),
    alpha=0.7,
    legend=None
)
axs[0].set_title('KernelBiome')
axs[0].set_xlabel('CFI')
axs[0].set_ylabel('')
axs[0].invert_yaxis()

cfi_and_gi_df.plot(
    x="label", y="ma_cfi_vals",
    kind="barh", ax=axs[1],
    color=tuple(colors_all[3]),
    alpha=0.7,
    legend=None
)
axs[1].set_title('KernelBiome-UF')
axs[1].set_xlabel('CFI')
axs[1].set_ylabel('')
axs[1].yaxis.set_ticklabels([])
axs[1].invert_yaxis()

cfi_and_gi_df.plot(
    x="label", y="pi_vals",
    kind="barh", ax=axs[2],
    color=tuple(colors_all[4]),
    alpha=0.7,
    legend=None
)
axs[2].set_title('RF')
axs[2].set_xlabel('Gini-Imp')
axs[2].set_ylabel('')
axs[2].yaxis.set_ticklabels([])
axs[2].invert_yaxis()

fig.set_size_inches(6, 10)
fig.savefig("output/cirrhotic_cfi_and_gi_full.pdf", bbox_inches="tight")

# %%
