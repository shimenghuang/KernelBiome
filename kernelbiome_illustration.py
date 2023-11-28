# %%
# libs
# ##
import pandas as pd
import numpy as np
from kernelbiome.kernelbiome import KernelBiome
from kernelbiome.kernels_jax import kmat_linear, kmat_aitchison
from kernelbiome.helpers_fitting import models_to_kernels
from helpers.load_data import load_processed
import matplotlib.pyplot as plt

# %%
# Load processed data and scale into simplex
# ##

data_name = "rmp"
data_path = "data_processed"
X, y, label_all, group_all = load_processed(data_name, data_path)
X /= X.sum(axis=1)[:, None]
print(X.shape)


# KernelBiome is a class and behaves similar to sklearn
# estimators. The most default call to KernelBiome is the following
# Note: this may take some time...


KB = KernelBiome(kernel_estimator='SVC',
                 center_kmat=True,
                 hyperpar_grid=None,
                 models=None,
                 cv_pars={},
                 estimator_pars={},
                 n_jobs=4,
                 random_state=None)

KB.fit(X, y, verbose=0)
pcs, cc = KB.kernelPCA(X, num_pc=2)
df = pd.DataFrame({'x1': pcs[:, 0],
                   'x2': pcs[:, 1],
                   'y': y})
groups = df.groupby('y')
for name, group in groups:
    plt.plot(group.x1, group.x2, marker='o', linestyle='',
             markersize=12, label=name)

plt.legend()
plt.show()
cfis = KB.compute_cfi(X)

# Fit on each model
bm = KB.best_models_
cfis_list = []
for ind in range(bm.shape[0]):
    model = {list(bm['estimator_key'])[ind]:
             list(bm['kmat_fun'])[ind]}
    KBrefit = KernelBiome(kernel_estimator='SVC',
                          center_kmat=True,
                          hyperpar_grid=None,
                          models=model,
                          cv_pars={},
                          estimator_pars={},
                          n_jobs=4,
                          random_state=None)
    KBrefit.fit(X, y)
    cfis = list(KBrefit.compute_cfi(X))
    cfis_list.append(cfis)

# Here the models/kernels (parameter 'models') were chosen based on
# our default list. Instead, one can also use one of the following
# ways to specify the kernels

# Option 1: A dictionary of models consisting of our default kernel
# names and a parameter dictionary
models = {
    'linear': None,
    'linear-weighted': None,
    'aitchison': {'c': np.logspace(-7, -3, 5)},
}

KB = KernelBiome(kernel_estimator='SVC',
                 models=models)
# Since one of the kernels is weighted we need to specify the weight
# matrix when fitting
KB.fit(X, y, w=np.eye(X.shape[1]))


# Option 2: A dictionary of kernel functions
kernel_dict = {
    'linear': kmat_linear,
    'non-standard-name': lambda x, y: kmat_aitchison(x, y,
                                                     c_X=0.0001,
                                                     c_Y=0.001)
}

KB = KernelBiome(kernel_estimator='SVC',
                 models=kernel_dict)
KB.fit(X, y)

# Option 3: One can used the function models_to_kernels to convert a
# dictonary of models to kernels (this is what happens internally in
# KernelBiome)
# Note: Here again we need to pass the weight matrix.
kernels = models_to_kernels(models, w=np.eye(X.shape[1]))
KB = KernelBiome(kernel_estimator='SVC',
                 models=kernels)
KB.fit(X, y)

# %%
# The parameters cv_pars and estimator_pars are dictionaries that can
# be used to pass parameters controlling the CV and estimators within
# KernelBiome

# The parameter hyperpar_grid can be used to specify an explicit
# hyperparameter grid. None corresponds to a default choice which is
# based on then eigenvalues of then kernel matrices.

# Once a model is fitted. One can get use the predict, predict_proba,
# compute_cfi, compute_cpd methods. The following toy example
# illustrates this:

# Simulated data from a toy model
n = 100
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)
X3 = np.random.normal(0, 1, n)
X4 = np.random.normal(0, 1, n)
X = np.exp(np.c_[X1, X2, X3, X4])
X /= X.sum(axis=1)[:, None]

y = 5*(X[:, 0]+X[:, 1])/(X[:, 0]+X[:, 1]+X[:, 2]) + np.random.normal(0, 1, n)/2

# Fit KernelBiome
KB = KernelBiome(kernel_estimator='KernelRidge',
                 center_kmat=True,
                 models=None)
KB.fit(X, y)
# MSE
MSE = np.sqrt(np.mean((KB.predict(X) - y)**2))
# Compute CFIs
cfi_unweighted = KB.compute_cfi(X)
# Compute CPDs and plot
cpd_values = KB.compute_cpd(X)
for j in range(4):
    plt.scatter(X[:, j], cpd_values[j, :])
plt.show()
# Compute kernelPCA and plot
pcs, expvar = KB.kernelPCA(X)
plt.scatter(pcs[:, 0], pcs[:, 1], c=y, cmap='Greens_r')
plt.show()


# Fit KernelBiome with weighted aitchison only
W = np.eye(4)
W[2, 0] = 1
W[0, 2] = 1
weighted_aitchison_mod = {
    'aitchison-weighted': {'c': [0]},
}

KB = KernelBiome(kernel_estimator='KernelRidge',
                 center_kmat=True,
                 models=weighted_aitchison_mod)
KB.fit(X, y, w=W)
# MSE
MSE = np.sqrt(np.mean((KB.predict(X) - y)**2))
# Compute CFIs
cfi_weighted = KB.compute_cfi(X)
# Compute CPDs and plot
cpd_values = KB.compute_cpd(X)
for j in range(4):
    plt.scatter(X[:, j], cpd_values[j, :])
plt.show()
# Compute kernelPCA and plot
pcs, expvar = KB.kernelPCA(X)
plt.scatter(pcs[:, 0], pcs[:, 1], c=y, cmap='Greens_r')
plt.show()
