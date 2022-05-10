# KernelBiome

Note: currently this repository contains code that can reproduce outputs in the paper [TODO: add link]. The `KernelBiome` python package is working-in-progress.

## Current Content

### `data`

This folder should contain two subfolders: 

- `MLRepo` which should contain a subfolder `qin2014` ([data source](https://github.com/knights-lab/MLRepo/tree/master/datasets/qin2014))
- `CentralParkSoil` should contain data pre-processed by `prep_centralparksoil.R` ([data source](https://github.com/jacobbien/trac-reproducible/tree/main/CentralParkSoil/original))

### `src`

- `metrics_jax.py`: 
- `kernels_jax.py`: 
- `helpers_jax.py`: 
- `weighted_kernels_jax.py`:
- `helpers_weighting.py`: 
- `tree_utils.py`:
- `cfi.py`:
- `utils.py`:

### `scripts`

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

- `workflow_demo.ipynb`: 

