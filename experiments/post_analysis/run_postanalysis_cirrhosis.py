# Note: add path so it also works for interactive run in vscode...
import sys  # nopep8
sys.path.insert(0, "../../")  # nopep8

from helpers.load_data import load_processed
from kernelbiome.kernelbiome import KernelBiome
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

###
# paths
###

# Print working directory
print(f'cwd: {os.getcwd()}')

if os.getcwd().endswith('experiments/post_analysis'):
    data_path = "../../data_processed"
    output_path = "../../experiments/post_analysis/results/"
else:
    # assuming running from kernelbiome_tmp
    data_path = "data_processed"
    output_path = "experiments/post_analysis/results/"
os.makedirs(output_path, exist_ok=True)


###
# Load cirrhosis data and preprocess
###

data_name = "cirrhosis"
X, y, label_all, group_all = load_processed(data_name, data_path)
disease_label = np.array(['Cirrhosis']*len(y))
disease_label[y == 1] = 'Healthy'
X /= X.sum(axis=1)[:, None]
print(X.shape)


###
# Screen to 50 taxa using KernelBiome and CFI
###

file_exists = os.path.exists(
    os.path.join(output_path, 'CFIscreen_cirrhosis.npy'))
if file_exists:
    cfis_screen = np.load(os.path.join(
        output_path, "CFIscreen_cirrhosis.npy"))
else:
    minX = X[X != 0].min()
    model = {'aitchison': {'c': [minX/2]}}

    # Fit Aitchison KB
    KB = KernelBiome(kernel_estimator='SVC',
                     center_kmat=True,
                     models=model,
                     estimator_pars={'n_hyper_grid': 40},
                     n_jobs=4)
    KB.fit(X, y)
    cfis_screen = KB.compute_cfi(X, verbose=1)
    np.save(os.path.join(output_path, "CFIscreen_cirrhosis.npy"),
            cfis_screen)

# Screen data
ind = np.argsort(np.abs(cfis_screen))[-50:]
X = X[:, ind]
X /= X.sum(axis=1)[:, None]
label_all = label_all[ind]
print(X.shape)


###
# Analysis based on KernelBiome - cirrhosis
###

# Fit full KernelBiome
KB = KernelBiome(kernel_estimator='SVC',
                 center_kmat=True,
                 estimator_pars={'cache_size': 1000,
                                 'n_hyper_grid': 40},
                 n_jobs=4)
KB.fit(X, y)

# Save best models
print(KB.best_models_[['estimator_key', 'avg_test_score']])
KB.best_models_.to_csv(os.path.join(output_path, 'bestmodels_cirrhosis.csv'),
                       index=False)


# Refit with a single kernel
ind = 0
model = {list(KB.best_models_['estimator_key'])[ind]:
         list(KB.best_models_['kmat_fun'])[ind]}
KB2 = KernelBiome(kernel_estimator='SVC',
                  center_kmat=True,
                  models=model,
                  estimator_pars={'cache_size': 1000,
                                  'n_hyper_grid': 40},
                  n_jobs=4)
KB2.fit(X, y)

# Summary statistics and diversity

# compute geometric median of healthy class in kernel distance
Xh = X[y == 1, :]
dist_vec = KB.kernelDist(Xh, Xh).sum(axis=0)
u = Xh[np.argmin(dist_vec), :]
df_diversity = pd.DataFrame({'kerneldist':
                             KB2.kernelSumStat(X, u=u),
                             'simpson-diversity':
                             1 - (X**2).sum(axis=1),
                             'disease': disease_label})
df_diversity.to_csv(
    os.path.join(output_path, 'diversity_cirrhosis.csv'),
    index=False)


# Kernel PCA
pcs, cc = KB2.kernelPCA(X, num_pc=2)
df = pd.DataFrame({'x1': pcs[:, 0],
                   'x2': pcs[:, 1],
                   'disease': disease_label})
df.to_csv(os.path.join(output_path, 'kernelPCA_cirrhosis.csv'),
          index=False)
np.save(os.path.join(output_path, "kernelPCA_cirrhosis.npy"), cc)

# CFI analysis
cfis = KB2.compute_cfi(X, verbose=1)
np.save(os.path.join(output_path, "CFI_cirrhosis.npy"), cfis)
