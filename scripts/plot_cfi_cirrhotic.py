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

short_label = [lbl.split(';')[-1] for lbl in label]
short_label = [lbl.split('__')[1] for lbl in short_label]
short_label = np.array([lbl.replace('_', ' ') for lbl in short_label])

# unweighted CFI
cirrhotic_cfi_vals = np.load("output/cirrhotic_unweighted_cfi_vals.npy")

# artificially weighted CFI
cirrhotic_art_cfi_vals = np.load(
    "output/cirrhotic_weighted_artificial_cfi_vals.npy")

# weighted CFI (M^A)
cirrhotic_ma_cfi_vals = np.load("output/cirrhotic_weighted_MA_cfi_vals.npy")

# %%
# Create the 3-subplot figure in main text
# ^^^^^^^

# Gini impurity feature importance
X_all = X_df.to_numpy().astype('float').T
X_all /= X_all.sum(axis=1)[:, None]
rf = RandomForestClassifier(
    max_depth=np.sqrt(X_all.shape[0]), random_state=seed_num)
rf.fit(X_all, y)
gi_vals = rf.feature_importances_

fig, axs = plt.subplots(1, 3,
                        gridspec_kw={'width_ratios': [2, 2, 1]},
                        constrained_layout=True)

# fig 1:
# 109 is a fake dimension as place holder in the plot
idx_keep = [102, 11,  56,  76,  96,  90,  13, 74, 108, 36]
short_label_fig1 = np.concatenate([short_label, ['...']])
cfi_vals_fig1 = np.concatenate([cirrhotic_cfi_vals, [0]])
cfi_vals_w1_fig1 = np.concatenate([cirrhotic_art_cfi_vals, [0]])
compare_cfi_df = pd.DataFrame({
    'label': short_label_fig1[idx_keep],
    'unweighted': cfi_vals_fig1[idx_keep],
    'weighted': cfi_vals_w1_fig1[idx_keep]
})

compare_cfi_df.plot(
    x="label", y=["unweighted", "weighted"],
    kind="barh", ax=axs[0],
    color=[tuple(c) for c in colors_all[:2]]*8,
    alpha=0.7
    # legend=None
)
axs[0].invert_yaxis()
# array(['Veillonella parvula', 'Streptococcus parasanguinis',
#        'Bacteroides stercoris', 'Lachnospiraceae bacterium 2 1 46FAA',
#        '[Clostridium] symbiosum', 'Erysipelatoclostridium ramosum',
#        'Ruminococcus bicirculans', 'Bacteroides uniformis'], dtype='<U35')

axs[0].set_yticklabels([
    "Veil.parv.", 'Bact.ster.', 'Lach.bact.', 'Clos.symb.', 'Erys.ramo.',
    'Rumi.bici.', 'Bact.unif.', 'Clos.citr.', r'$\cdots$', 'Stre.para.'
], ha='center')
axs[0].get_yticklabels()[0].set_color("red")
axs[0].get_yticklabels()[9].set_color("red")
axs[0].tick_params(axis='y', which='major', pad=17)
axs[0].set_ylabel('')
axs[0].set_xlabel('CFI')
axs[0].legend(bbox_to_anchor=(-0.05, 1.05, 1.1, .1),
              loc='lower left', handletextpad=0.2,
              ncol=2, mode="expand", borderaxespad=0.1,
              prop={'size': 5.5})

# for ii in [0, 10, 9, 19]:
#     patch = axs[0].patches[ii]
#     patch.set_edgecolor('black')
#     patch.set_alpha(1.0)

# fig 2
n_comp = 10
idx_keep = np.argsort(-np.abs(cirrhotic_ma_cfi_vals))[:n_comp]
cfi_gi_df = pd.DataFrame({
    'label': short_label[idx_keep],
    'CFI (UW)': cirrhotic_ma_cfi_vals[idx_keep],
    'GI': gi_vals[idx_keep]
})

cfi_gi_df.plot(
    x="label", y='CFI (UW)', kind="barh", ax=axs[1],
    color=tuple(colors_all[3]), alpha=0.8,
    legend=None)
# array(['Veillonella parvula', 'Streptococcus parasanguinis',
#        'Streptococcus mitis', 'Sutterella wadsworthensis',
#        'Streptococcus salivarius', 'Phascolarctobacterium succinatutens',
#        'Fusobacterium mortiferum', 'Lachnospiraceae bacterium 2 1 46FAA'],
#       dtype='<U35')
axs[1].set_yticklabels([
    'Veil.parv.', 'Stre.para.', 'Stre.miti.', 'Sutt.wads.', 'Stre.sali.',
    'Phas.succ.', 'Fuso.mort.', 'Lach.bact.', 'Para.excr.', 'Euba.sira.'
], ha='center')
axs[1].tick_params(axis='y', which='major', pad=17)
axs[1].invert_yaxis()
axs[1].set_title('KernelBiome-UF', fontsize=6.5)
axs[1].set_xlabel('CFI')
axs[1].set_ylabel('')

cfi_gi_df.plot(
    x="label", y='GI', kind="barh", ax=axs[2],
    color=tuple(colors_all[4]), alpha=0.8,
    legend=None)

axs[2].invert_yaxis()
axs[2].set_yticklabels([])
axs[2].tick_params(left=False)
axs[2].set_title('RF', fontsize=6.5)
axs[2].set_xlabel('Gini-Imp')
axs[2].set_ylabel('')

# handles, labels = [(a + b) for a, b in zip(axs[0].get_legend_handles_labels(),
#                                            axs[1].get_legend_handles_labels())]
# fig.legend(handles, labels, bbox_to_anchor=(0.4, -0.1, 0.5, 0.5))

# plt.subplots_adjust(wspace=0.4)
# fig.tight_layout()

plt.style.use("seaborn-white")
# fig.tight_layout()
for spine in axs[0].spines.values():
    spine.set(color='gray', linewidth=2, alpha=0.2)
for spine in axs[1].spines.values():
    spine.set(color='gray', linewidth=2, alpha=0.2)
for spine in axs[2].spines.values():
    spine.set(color='gray', linewidth=2, alpha=0.2)
plt.style.use("seaborn-white")
fig.set_size_inches(4, 1.9)
fig.savefig("output/cirrhotic_cfi_and_gi.pdf",
            bbox_inches='tight')

# %%
# Create a 3-subplot figure for the appendix
# ^^^^^^

selected_comp = np.argsort(-np.abs(cirrhotic_cfi_vals))

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

plt.style.use("seaborn-white")
fig.set_size_inches(6, 10)
fig.savefig("output/cirrhotic_cfi_and_gi_full.pdf", bbox_inches="tight")

# %%
