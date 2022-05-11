# KernelBiome

Note: currently this repository contains code that can reproduce outputs in the paper [TODO: add link]. The `KernelBiome` python package will be available in the upcoming two weeks.

## Current Content

### `data`

This folder should contain two subfolders: 

- `MLRepo` which should contain a subfolder `qin2014` containing data directly taken from [here](https://github.com/knights-lab/MLRepo/tree/master/datasets/qin2014).
- `CentralParkSoil` should contain data pre-processed by `prep_centralparksoil.R` based on data from [here](https://github.com/jacobbien/trac-reproducible/tree/main/CentralParkSoil/original).

### `src`

This folder contains most functions to be included in the `KernelBiome` package:

- `metrics_jax.py`: 
- `kernels_jax.py`: 
- `helpers_jax.py`: 
- `weighted_kernels_jax.py`:
- `helpers_weighting.py`: 
- `tree_utils.py`:
- `cfi.py`:
- `utils.py`:
- `simplex_heatmap.py`:

### `scripts`

This folder contains scripts that can reproduce resutls in the paper:

- `load_<ds>.py`: 
- `save_cv_indices_<ds>.py`:
- `create_unifrac_weights_<ds>.py`:
- `setup_params.py`:
- `run_<ds>_compare.py`:
- `run_<ds>_cfi.py`:
- `plot_<ds>_cfi.py`:
- `plot_<ds>_cpd.py`:
- `plot_cirrhotic_mds.py`:

### `notebooks`

This folder contains notebooks for demonstration:

- `workflow_demo.ipynb`: A workflow demonstration using simulated data.
- `visualization_writeup.ipynb`: Visualization of kernels with heatmap on 2-simplex.


