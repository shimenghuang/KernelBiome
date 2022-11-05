# %%
# libs
# ^^^^^^

# Note: add path so it also works for interactive run in vscode...
import sys  # nopep8
sys.path.insert(0, "../../")  # nopep8


import os
import numpy as np
from kernelbiome.kernelbiome import KernelBiome
from helpers.load_data import load_processed
from helpers.draw_tree_utils import create_node_df
from helpers.unifrac import compute_unifrac
import matplotlib.pyplot as plt


# Print working directory
print(os.getcwd())

data_path = "/Users/elisabeth.ailer/Projects/P3_Kernelbiome/kernelbiome_tmp/data_processed"
output_path = "/Users/elisabeth.ailer/Projects/P3_Kernelbiome/kernelbiome_tmp/experiments/tree_visualization/results"
os.makedirs(output_path, exist_ok=True)


# --------------------------------------------------------------------------------------------------------------
# Load cirrhosis data and preprocess
# --------------------------------------------------------------------------------------------------------------

data_name = "centralparksoil"
X, y, label_all, group_all = load_processed(data_name, data_path)
X /= X.sum(axis=1)[:, None]
print(X.shape)


# --------------------------------------------------------------------------------------------------------------
# Screen to 50 taxa using KernelBiome and CFI
# --------------------------------------------------------------------------------------------------------------
minX = X[X != 0].min()
model = {'aitchison': {'c': [minX/2]}}

if data_name == "cirrhosis":
    # Fit Aitchison KB
    KB = KernelBiome(kernel_estimator='SVC',
                     center_kmat=True,
                     models=model,
                     estimator_pars={'cache_size': 1000,
                                     'n_hyper_grid': 40},
                     n_jobs=4)

elif data_name == "centralparksoil":
    # Fit Aitchison KB
    KB = KernelBiome(kernel_estimator='KernelRidge',
                     center_kmat=True,
                     models=model,
                     cv_pars={'outer_cv_type': "kfold"},
                     n_jobs=4)

KB.fit(X, y)
cfis_screen = KB.compute_cfi(X, verbose=1)

# Screen data (keep top 50)
ind = np.argsort(np.abs(cfis_screen))[-50:]
X = X[:, ind]
X /= X.sum(axis=1)[:, None]
label_all = label_all[ind]
print(X.shape)
node_df = create_node_df(label_all)

minX = X[X != 0].min()
if data_name == "cirrhosis":
    models = {'aitchison': {'c': [minX/2]},
              'generalized-js': {'ab': [[1, 0.5]]}}
elif data_name == "centralparksoil":
    models = {'aitchison': {'c': [minX / 2]},
              'generalized-js': {'ab': [[1, 0.5]]},
              'aitchison-rbf': {'cg': [[0.00031, 0.0005]]}
              }



# --------------------------------------------------------------------------------------------------------------
# Compute unifrac weights
# --------------------------------------------------------------------------------------------------------------
unifrac_path = os.path.join(output_path, f"{data_name}_weights")
os.makedirs(output_path, exist_ok=True)

biom_file = os.path.join(unifrac_path, f"{data_name}_1sample.biom")
tree_file = os.path.join(unifrac_path, f"{data_name}_tree.tre")

levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
delim = ";"


# Compute dummy weights
n_taxa = len(label_all)
w_dummy = np.ones((n_taxa, n_taxa))
phylum_vec = node_df["Phylum"].values
for i in np.arange(n_taxa):
    for j in np.arange(n_taxa):
        if phylum_vec[i] != phylum_vec[j]:
            w_dummy[i, j] = 0
#w_dummy = w_dummy / w_dummy.sum(axis=1)

np.save(os.path.join(unifrac_path, f"{data_name}_w_dummy.npy"), w_dummy)


# W^{unifrac} with M^A
res = compute_unifrac(label_all, biom_file, tree_file,
                      set_branch_length=True,
                      branch_length=1.0,
                      levels=levels,
                      delim=delim,
                      table_id="Dataset")
dist_unifrac = res["UnifracTable"].data
M = 1.0-dist_unifrac
D = np.diag(1.0/np.sqrt(np.diag(M)))
W_MA = D.dot(M).dot(D)
np.save(os.path.join(unifrac_path, f"{data_name}_w_unifrac_MA.npy"), W_MA)

# W^{unifrac} with M^B
M = dist_unifrac.T.dot(dist_unifrac)
D = np.diag(1.0/np.sqrt(np.diag(M)))
W_MB = D.dot(M).dot(D)
np.save(os.path.join(unifrac_path, f"{data_name}_w_unifrac_MB.npy"), W_MB)

plt.imshow(W_MA)
plt.xlabel("Species")
plt.ylabel("Species")
plt.savefig(os.path.join(unifrac_path, f"{data_name}_w_unifrac_MA_plot.pdf"))

plt.imshow(W_MB)
plt.xlabel("Species")
plt.ylabel("Species")
plt.savefig(os.path.join(unifrac_path, f"{data_name}_w_unifrac_MB_plot.pdf"))

plt.imshow(w_dummy)
plt.xlabel("Species")
plt.ylabel("Species")
plt.savefig(os.path.join(unifrac_path, f"{data_name}_w_unifrac_dummy_plot.pdf"))



# --------------------------------------------------------------------------------------------------------------
# Compute CFI values for weighted/unweighted Kernelbiome
# --------------------------------------------------------------------------------------------------------------

for mod, par in models.items():
    # Unweighted KernelBiome
    model = {mod: par}

    if data_name == "cirrhosis":
        KB = KernelBiome(kernel_estimator='SVC',
                         center_kmat=True,
                         models=model,
                         estimator_pars={'cache_size': 1000,
                                         'n_hyper_grid': 40},
                         n_jobs=4)
    elif data_name == "centralparksoil":
        KB = KernelBiome(kernel_estimator='KernelRidge',
                         center_kmat=True,
                         models=model,
                         cv_pars={'outer_cv_type': "kfold"},
                         n_jobs=4)
    # Unweighted CFI
    KB.fit(X, y)
    cfi_unweighted = KB.compute_cfi(X)
    node_df[f'cfi_unweighted_{mod}'] = cfi_unweighted
    # Weighted KernelBiome
    model = {mod+'-weighted': par}

    if data_name == "cirrhosis":
        KB = KernelBiome(kernel_estimator='SVC',
                         center_kmat=True,
                         models=model,
                         estimator_pars={'cache_size': 1000,
                                         'n_hyper_grid': 40},
                         n_jobs=4)
    elif data_name == "centralparksoil":
        KB = KernelBiome(kernel_estimator='KernelRidge',
                         center_kmat=True,
                         models=model,
                         cv_pars={'outer_cv_type': "kfold"},
                         n_jobs=4)
    # Dummy weighted CFI
    KB.fit(X, y, w=w_dummy)
    cfi_dummy = KB.compute_cfi(X)
    node_df[f'cfi_dummy_{mod}'] = cfi_dummy

    # Unifrac weighted CFI W_MA
    KB.fit(X, y, w=W_MA)
    cfi_unifrac_WMA = KB.compute_cfi(X)
    node_df[f'cfi_unifrac_wma_{mod}'] = cfi_unifrac_WMA

    # Unifrac weighted CFI W_MB
    KB.fit(X, y, w=W_MB)
    cfi_unifrac_WMB = KB.compute_cfi(X)
    node_df[f'cfi_unifrac_wmb_{mod}'] = cfi_unifrac_WMB

# Save results
node_df.to_csv(os.path.join(output_path, f"{data_name}_weights", 'CFIandLabels.csv'))

