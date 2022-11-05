# Note: add path so it also works for interactive run in vscode...
import sys  # nopep8
sys.path.insert(0, "../../")  # nopep8

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

from kernelbiome.kernelbiome import KernelBiome
from helpers.load_data import load_processed

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
# Load centralparksoil data and preprocess
###

data_name = "centralparksoil"
X, y, label_all, group_all = load_processed(data_name, data_path)
X /= X.sum(axis=1)[:, None]
print(X.shape)

###
# Screen to 50 taxa using KernelBiome and CFI
###

file_exists = os.path.exists(
    os.path.join(output_path, 'CFIscreen_centralparksoil.npy'))
if file_exists:
    cfis_screen = np.load(os.path.join(
        output_path, "CFIscreen_centralparksoil.npy"))
else:
    minX = X[X != 0].min()
    model = {'aitchison': {'c': [minX/2]}}

    # Fit Aitchison KB
    KB = KernelBiome(kernel_estimator='KernelRidge',
                     center_kmat=True,
                     models=model,
                     estimator_pars={'n_hyper_grid': 40},
                     n_jobs=4)
    KB.fit(X, y)
    cfis_screen = KB.compute_cfi(X, verbose=1)
    np.save(os.path.join(output_path, "CFIscreen_centralparksoil.npy"),
            cfis_screen)

# Screen data
ind = np.argsort(np.abs(cfis_screen))[-50:]
X = X[:, ind]
X /= X.sum(axis=1)[:, None]
label_all = label_all[ind]
print(X.shape)


###
# Analysis based on KernelBiome - centralparksoil
###

# Fit full KernelBiome
KB = KernelBiome(kernel_estimator='KernelRidge',
                 center_kmat=True,
                 estimator_pars={'n_hyper_grid': 40},
                 n_jobs=4)
KB.fit(X, y)

# Save best models
print(KB.best_models_[['estimator_key', 'avg_test_score']])
KB.best_models_.to_csv(os.path.join(output_path,
                                    'bestmodels_centralparksoil.csv'),
                       index=False)


# Refit with a single kernel
ind = 0
model = {list(KB.best_models_['estimator_key'])[ind]:
         list(KB.best_models_['kmat_fun'])[ind]}
KB2 = KernelBiome(kernel_estimator='KernelRidge',
                  center_kmat=True,
                  models=model,
                  estimator_pars={'n_hyper_grid': 40},
                  n_jobs=4)
KB2.fit(X, y)

# Kernel PCA
pcs, cc = KB2.kernelPCA(X, num_pc=2)
df = pd.DataFrame({'x1': pcs[:, 0],
                   'x2': pcs[:, 1]})
df.to_csv(os.path.join(output_path, 'kernelPCA_centralparksoil.csv'),
          index=False)
np.save(os.path.join(output_path, "kernelPCA_centralparksoil.npy"), cc)

# CFI analysis
cfis = KB2.compute_cfi(X, verbose=1)
np.save(os.path.join(output_path, "CFI_centralparksoil.npy"), cfis)
