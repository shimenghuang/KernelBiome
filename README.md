# KernelBiome

Note: This repository contains code that can reproduce outputs in the paper [Supervised Learning and Model Analysis with Compositional Data (Huang et al., 2022)](https://arxiv.org/abs/2205.07271). The `KernelBiome` python package can be installed via 

```
pip install kernelbiome
```
or
```
python -m pip install git+https://github.com/shimenghuang/KernelBiome.git
```

Minimum usage example for testing out installation:

```
import numpy as np
from kernelbiome.kernels_jax import *

x = np.random.uniform(0, 1, 5)
x /= x.sum()
y = np.random.uniform(0, 1, 5)
y /= y.sum()

k_linear(x,y)
```

## The KernelBiome Package

The package contains the following modules:

- `metrics_jax.py`: distance metrics with jax.
- `kernels_jax.py`: kernel functions with jax.
- `helpers_jax.py`: helper functions for kernels and metrics with jax.
- `weighted_kernels_jax.py`: weighted kernel functions with jax.
- `helpers_weighting.py`: helpers for calculating weight matrices. 
- `cfi_and_cpd.py`: CFI and CPD calculation.
- `nested_cv`: functions for nested CV.
- `utils_cv.py`: utility functions including utilities for nested CV 
- `utils_result.py`: utility functions for result summary.

## Reproducible Code

### `data`

This folder should contain two subfolders: 

- `MLRepo` which should contain a subfolder `qin2014` containing data directly taken from [here](https://github.com/knights-lab/MLRepo/tree/master/datasets/qin2014).
- `CentralParkSoil` should contain data pre-processed by `prep_centralparksoil.R` based on data from [here](https://github.com/jacobbien/trac-reproducible/tree/main/CentralParkSoil/original).

### `scripts`

This folder contains scripts that can reproduce resutls in the paper:

- `tree_utils.py`: utility functions regardign the UniFrac distance.
- `load_<ds>.py`: load the dataset into proper format.
- `create_unifrac_weights_<ds>.py`: create the UniFrac based weight matrices.
- `setup_params.py`: set up kernel parameters and hyperparameters.
- `run_cfi_cpd_<ds>.py`: calculate CFI and CPD.
- `plot_<ds>_cfi.py`: make CFI plot.
- `plot_<ds>_cpd.py`: make CPD plot.
- `plot_mds_cirrhotic.py`: make MDS plot.
- `save_cv_indices_<ds>.py`: save the CV indices for the 50-fold CV comparison so that each approach is run on the same part of the data in `run_compare_one_fold_<ds>.py`.
- `run_compare_one_fold_<ds>.py`: run one of the 50-fold CV using different methods.

where `<ds>` is one of `cirrhotic` or `centralparksoil`.

### `notebooks`

This folder contains notebooks for demonstration:

- `simplex_heatmap.py`: heatmap functons to visualize kernels on 2-simplex.
- `workflow_demo.ipynb`: A workflow demonstration using simulated data.
- `visualization_writeup.ipynb`: Visualization of kernels with heatmap on 2-simplex.
