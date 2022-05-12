# %%
# load libs
# ^^^^^^

import sys  # nopep8
sys.path.insert(0, "../")  # nopep8

from matplotlib import rc
import matplotlib.pyplot as plt
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

# Unweighted CPD
cirrhotic_cpd_vals = np.load("output/cirrhotic_unweighted_cpd_vals.npy")

# Weighted CPD (M^A)
cirrhotic_ma_cpd_vals = np.load("output/cirrhotic_weighted_MA_cpd_vals.npy")

# %%
# plot unweighted CPD and CFI (top 10 according to CFI)
# ^^^^^^

selected_comp_unweighted = np.argsort(-np.abs(cirrhotic_cfi_vals))[:10]
selected_comp_weighted = np.argsort(-np.abs(cirrhotic_ma_cfi_vals))[:10]

short_label = [lbl.split(';')[-1] for lbl in label]
short_label = [lbl.split('__')[1] for lbl in short_label]
short_label = np.array([lbl.replace('_', ' ') for lbl in short_label])

X_all = X_df.to_numpy().astype('float').T
X_all /= X_all.sum(axis=1)[:, None]

abbr_label = ['Veil.parv.', 'Bact.ster.', 'Lach.bact.', 'Clos.symb.', 'Erys.ramo.',
              'Rumi.bici.', 'Bact.unif.', 'Clos.citr.', 'Blau.hans.', 'Prev.timo.']

plt.style.use("seaborn-white")
fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
plot_cfi(X_all[:, selected_comp_unweighted], cirrhotic_cpd_vals[selected_comp_unweighted], labels=[
         f'$X^{jj+1}$' for jj in range(X_all.shape[1])], colors=colors_all[::2], axs=axs[0], xlabel='Relative abundance')
# axs[0].set_xlim((0,0.05))
axs[0].annotate('Veil.parv.', (0.45, -0.38))
axs[0].annotate('Bact.ster.', (0.3, 0.1))
axs[0].annotate('Lach.bact.', (0.02, 0.12))
axs[0].annotate('Clos.symb.', (0.015, -0.05))
axs[0].annotate('Erys.ramo.', (0.01, -0.13))
axs[0].annotate('Rumi.bici.', (0.11, -0.1))
axs[0].annotate('Bact.unif.', (0.31, 0.05))
axs[0].annotate('Clos.citr.', (0.01, 0.05))
axs[0].annotate('Prev.timo.', (0.01, 0.03))
axs[0].annotate('Blau.hans.', (0.01, 0.01))
plot_cfi_index(cirrhotic_cfi_vals[selected_comp_unweighted], fmt='%.2f',
               labels=abbr_label, colors=colors_all[::2], axs=axs[1], ascending=False)
fig.set_size_inches(3.2, 2.5)
fig.savefig('output/cirrhotic_unweighted_cpd_and_cfi_top10.pdf',
            bbox_inches='tight')


# %%
# plot unweighted CPD and CFI (top 10 according to CFI)
# ^^^^^^

abbr_label = ['Veil.parv.', 'Stre.para.', 'Stre.miti.', 'Sutt.wads.', 'Stre.sali.',
              'Phas.succ.', 'Fuso.mort.', 'Lach.bact.', 'Para.exre.', 'Euba.sira.']

plt.style.use("seaborn-white")
fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
plot_cfi(X_all[:, selected_comp_weighted], cirrhotic_ma_cpd_vals[selected_comp_weighted], labels=[
         f'$X^{jj+1}$' for jj in range(X_all.shape[1])], colors=colors_all[::2], axs=axs[0], xlabel='Relative abundance')
# axs[0].set_xlim((0,0.05))
axs[0].annotate('Veil.parv.', (0.42, -0.35))
axs[0].annotate('Stre.para.', (0.06, -0.14))
axs[0].annotate('Stre.miti.', (-0.01, 0.08))
axs[0].annotate('Sutt.wads.', (0.05, 0.12))
axs[0].annotate('Stre.sali.', (0.3, -0.08))
axs[0].annotate('Phas.succ.', (0.15, -0.05))
axs[0].annotate('Fuso.mort.', (0.31, -0.15))
axs[0].annotate('Lach.bact.', (0.02, 0.04))
axs[0].annotate('Para.exre.', (0.025, 0.02))
axs[0].annotate('Euba.sira.', (0.08, 0.06))

plot_cfi_index(cirrhotic_ma_cfi_vals[selected_comp_weighted], fmt='%.2f',
               labels=abbr_label, colors=colors_all[::2], axs=axs[1], ascending=False)
fig.set_size_inches(3.2, 2.5)
fig.savefig('output/cirrhotic_weighted_MA_cpd_and_cfi_top10.pdf',
            bbox_inches='tight')

# %%

# %%
