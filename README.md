# KernelBiome

Note: currently this repository contains code that can reproduce outputs in the paper [TODO: add link]. The `KernelBiome` python package will be available within the upcoming two weeks.

## Current Content

### `data`

This folder should contain two subfolders: 

- `MLRepo` which should contain a subfolder `qin2014` containing data directly taken from [here](https://github.com/knights-lab/MLRepo/tree/master/datasets/qin2014).
- `CentralParkSoil` should contain data pre-processed by `prep_centralparksoil.R` based on data from [here](https://github.com/jacobbien/trac-reproducible/tree/main/CentralParkSoil/original).

### `src`

This folder contains most functions to be included in the `KernelBiome` package:

- `metrics_jax.py`: distance metrics with jax.
- `kernels_jax.py`: kernel functions with jax.
- `helpers_jax.py`: helper functions for kernels and metrics with jax.
- `weighted_kernels_jax.py`: weighted kernel functions with jax.
- `helpers_weighting.py`: helpers for calculating weight matrices. 
- `tree_utils.py`: utility functions regardign the UniFrac distance.
- `cfi.py`: CFI and CPD calculation.
- `utils.py`: general utility functions including utilities for nested CV and result summary.
- `simplex_heatmap.py`: heatmap functons to visualize kernels on 2-simplex.

### `scripts`

This folder contains scripts that can reproduce resutls in the paper:

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

- `workflow_demo.ipynb`: A workflow demonstration using simulated data.
- `visualization_writeup.ipynb`: Visualization of kernels with heatmap on 2-simplex.
