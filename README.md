# KernelBiome package

The `KernelBiome` python package can be installed via 

```
pip install kernelbiome
```
or
```
python -m pip install git+https://github.com/shimenghuang/KernelBiome.git
```

Small usage example:

```
import numpy as np
from kernelbiome.kernelbiome import KernelBiome

# Simulated some data
n = 100
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)
X3 = np.random.normal(0, 1, n)
X4 = np.random.normal(0, 1, n)
X = np.exp(np.c_[X1, X2, X3, X4])
X /= X.sum(axis=1)[:, None]
y = 5*(X[:, 0]+X[:, 1])/(X[:, 0]+X[:, 1]+X[:, 2]) + np.random.normal(0, 1, n)/2

# Fit KernelBiome
models = {
    'linear': None,
    'aitchison': {'c': np.logspace(-7, -3, 5)},
}
KB = KernelBiome(kernel_estimator='KernelRidge',
                 center_kmat=True,
                 models=models, # `models=None` for using all default models
                 verbose=1)
KB.fit(X, y)

# Calculate mean squared error
MSE = np.sqrt(np.mean((KB.predict(X) - y)**2))
```

For a complete usage example, see `kernelbiome_illustration.py`


# Reproducible Code

This [repository](https://github.com/shimenghuang/KernelBiome) contains the python package `KernelBiome` and code that can reproduce results in the paper [Supervised Learning and Model Analysis with Compositional Data (Huang et al., 2022)](https://arxiv.org/abs/2205.07271). 

All scripts producing results in the paper can be found in the `experiments` folder with some helper functions for the experiment scripts located in the `helpers` folder. Scripts starting with "run_" are used to run computation and save results, and scripts starting with "summarize_" are used to load and summarize results in e.g. figures. `data_original` and `data_processed` are folder to place the original and to save the processed datasets respectively. See README files therein for details.

## `prediction`

Prediction comparison on the 33 publicly available datasets on classification and regression.

## `post_analysis`

Post-analysis including CFI and kernel PCA for two of the public datasets, `cirrhosis` and `centralpark`.

## `tree_visualization`

Visualization of CFI base on weighted and unweighted KernelBiome.

## `consistency`

Simulation to show consistency results in the paper.

### `toy_examples`

`log_contrast_example.py`: Illustration of CFI and CPD in the case of log contrast model using simulated data.

`rescale_matters_example.py`: Comparison of CFI and CPD with relative influence (RI) and partial dependency plot (PDP) based on simulated data.
