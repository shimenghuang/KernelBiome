# %%
# load libs
# ^^^^^^

import sys  # nopep8
sys.path.insert(0, "../")  # nopep8

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.metrics_jax import *
from src.cfi import *
from src.utils import *
import load_cirrhotic

seed_num = 2022

rc('font', **{'family': 'tex-gyre-termes', 'size': 6.5})
rc('text', usetex=True)
rc('text.latex',
   preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')

colors_all = np.vstack([plt.cm.tab20c(range(20)), plt.cm.tab20b(range(20))])
colors_all = np.unique(colors_all, axis=0)
print(colors_all.shape)

# %%
# load resutls
# ^^^^^^^

data_path = "/Users/hrt620/Documents/projects/kernelbiome_proj/kernelbiome_clean/data/MLRepo/qin2014"
X_df, y, label = load_cirrhotic.main(data_path=data_path, seed_num=seed_num)

X_all = X_df.to_numpy().astype('float').T
X_all /= X_all.sum(axis=1)[:, None]

param_grid_svm = dict(C=[10**x for x in [-3, -2, -1, 0, 1, 2, 3]])
print(param_grid_svm)

# %%
# refit best model
# ^^^^^^

# Note: taken from saved model selection results
# e.g. cirrhotic_unweighted_best_models_May-08-2022.csv
model_selected = pd.Series({
    'estimator_key': 'aitchison_c_1e-05',
    'kmat_fun': wrap(kmat_aitchison, c_X=1e-5, c_Y=1e-5)
})
print(model_selected)
pred_fun, gscv = refit_best_model(X_all, y, 'SVC', param_grid_svm, model_selected,
                                  'accuracy', center_kmat=False, n_fold=5, n_jobs=6, verbose=0)
print(gscv.best_estimator_)
print(f"* Done {model_selected.estimator_key}.")

# %%
# Kernel PCA + diversity plots
# ^^^^^^


def simpsons_diversity(x):
    return 1-(x**2).sum()


colors = [tuple(c) for c in colors_all[[8, 9]]]
labels = ['Healthy', 'Cirrhosis']

fig, axs = plt.subplots(1, 2,
                        gridspec_kw={'width_ratios': [1.8, 1]},
                        # constrained_layout=True,
                        figsize=(5, 4))

ms = ['o', '^']
Z1 = Phi(X_all, X_all, model_selected.kmat_fun,
         center=True, pc=0, return_mean=False)
Z2 = Phi(X_all, X_all, model_selected.kmat_fun,
         center=True, pc=1, return_mean=False)
for jj in range(2):
    cur_loc = y == 1 if jj == 0 else y == -1
    axs[0].scatter(Z1[cur_loc], Z2[cur_loc], s=3, marker=ms[jj],
                   #    c=np.repeat(colors[jj], sum(cur_loc)),
                   c=colors[jj],
                   label=labels[jj])

axs[0].legend(bbox_to_anchor=(0.05, 1., 0.8, .1),
              loc='lower left', handletextpad=-0.1,
              ncol=2, mode="expand", borderaxespad=-0.1)
axs[0].set_xlabel('First component', labelpad=0.5)
axs[0].set_ylabel('Second component', labelpad=0.2)
for spine in axs[0].spines.values():
    spine.set(color='gray', linewidth=2, alpha=0.2)

# diversity
u = np.repeat(1/X_all.shape[1], X_all.shape[1])
diversity = np.zeros(X_all.shape[0])
for ii in range(X_all.shape[0]):
    diversity[ii] = -d2_aitchison(X_all[ii], u, c=1e-5)

diversity_lst = [diversity[y == 1], diversity[y == -1]]
p1 = axs[1].boxplot(diversity_lst, positions=[1-0.15, 2-0.15], notch=True, widths=0.25, patch_artist=True,
                    boxprops=dict(
    facecolor=colors_all[0],
    edgecolor='white'),
    flierprops=dict(markersize=2))
axs[1].set_ylabel('Kernel diversity', labelpad=0.5, color=colors_all[0])

diversity2 = np.array([simpsons_diversity(x) for x in X_all])
diversity2_lst = [diversity2[y == 1], diversity2[y == -1]]
axs2 = axs[1].twinx()
p2 = axs2.boxplot(diversity2_lst, positions=[1+0.15, 2+0.15], notch=True, widths=0.25, patch_artist=True,
                  boxprops=dict(
    facecolor=colors_all[4], edgecolor='white'),
    flierprops=dict(markersize=2))
axs2.set_ylabel('Gini-Simpson diversity', labelpad=4, color=colors_all[4])
p1['medians'][0].set_color('black')
p1['medians'][1].set_color('black')
p2['medians'][0].set_color('black')
p2['medians'][1].set_color('black')

axs[1].set_xticks([1, 2])
axs[1].set_xticklabels(['Healthy', 'Cirrhosis'])
for spine in axs[1].spines.values():
    spine.set(color='gray', linewidth=2, alpha=0.2)
for spine in axs2.spines.values():
    spine.set_visible(False)

plt.subplots_adjust(wspace=0.5)
plt.style.use("seaborn-white")
fig.set_size_inches(3.45, 1.5)
fig.savefig("output/cirrhotic_mds_and_diversity.pdf", bbox_inches='tight')
plt.show()

# %%
