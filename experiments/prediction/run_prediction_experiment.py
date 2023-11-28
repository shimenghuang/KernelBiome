# %%
# path for interactive run
# ^^
from pathlib import Path  # nopep8
import sys  # nopep8
path_root = Path(__file__).parents[2]  # nopep8
sys.path.append(str(path_root))  # nopep8
print(sys.path)  # nopep8

# %%
# Imports
# ^^
from helpers.load_data import load_processed
from helpers.one_fold import (run_bl, run_svm_rbf,
                              run_lr_l1, run_classo,
                              run_rf, run_kb)
import os
from os.path import join
from pathlib import Path
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import (StratifiedKFold, LeaveOneOut,
                                     KFold)

# %%
# Add project to path
# ^^
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

# Local imports

# Print working directory
print(os.getcwd())

# Parse inputs
parser = argparse.ArgumentParser(
    description="Run comparison CV folds",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--nfold", type=int,
                    help="number of comparison CV fold")
parser.add_argument("-l", "--cvtype", type=str,
                    help="kfold for K-fold-CV and loo for LOO-CV")
parser.add_argument("-s", "--seed", type=int, help="seed number")
parser.add_argument("-d", "--dataname", help="name for the data")
parser.add_argument("-n", "--runname", help="name for the run")
parser.add_argument("-w", "--njob", type=int,
                    default=-1, help="number of jobs")
parser.add_argument("-i", "--startfold", type=int,
                    help="start from a particular fold i", default=0)
args = parser.parse_args()
config = vars(args)
print(config)

data_name = config['dataname']
num_fold = config['nfold']
loo = True if config['cvtype'] == "loo" else False
seed_num = config['seed']
run_name = config['runname']
n_jobs = config['njob']
startfold = config['startfold']
endfold = startfold + 1

# # Note: can use this if run interactively
# data_name = "centralparksoil"
# num_fold = 10
# seed_num = 10
# run_name = 'halo'
# startfold = 0
# loo = False
# endfold = num_fold - 1
# n_jobs = 1

# Input and output paths
data_path = "data_processed/"
output_path = f"experiments/prediction/results/{data_name}_{run_name}/"
os.makedirs(output_path, exist_ok=True)

# Save run config
np.save(join(output_path, "config.npy"), config)

# Load data and scale to simplex
X_all, y_all, label_all, group_all = load_processed(data_name, data_path)
X_all /= X_all.sum(axis=1)[:, None]
print(X_all.shape)

# Setup parameters for experiment
task_type = 'classification' if len(np.unique(y_all)) == 2 else 'regression'
if task_type == 'classification':
    scoring = "accuracy"
    if loo:
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(num_fold, shuffle=True, random_state=seed_num)
else:
    scoring = "neg_mean_squared_error"
    if loo:
        cv = LeaveOneOut()
    else:
        cv = KFold(num_fold, shuffle=True, random_state=seed_num)

# Generate CV splits
cv_idx_dict_all = []
for tr, te in cv.split(X_all, y_all):
    cv_idx_dict = {'tr': tr, 'te': te}
    cv_idx_dict_all.append(cv_idx_dict)
# Save CV splits
np.save(
    join(output_path, f'cv_idx_all_seed{seed_num}'), cv_idx_dict_all)

# Create list of models for Aitchison
aitchison_models = {'aitchison': {'c': np.logspace(-7, -3, 5)}}
minX = X_all[X_all != 0].min()
aitchison_models_single = {'aitchison': {'c': [minX/2]}}


# Specify hyperparameter grid for competitor methods
# Hyperparameter for Lin/Log-L1
if task_type == 'classification':
    param_grid_lr = dict(C=[10**x for x in [-4, -3, -2, -1, 0, 1, 2]])
else:
    param_grid_lr = dict(alpha=[1/(2*10**x) for x in [-4, -3, -2, -1, 0, 2]])
# Hyperparameter for SVM-RBF
param_grid_svm_rbf = dict(C=[10**x for x in [-3, -2, -1, 0, 1, 2]],
                          gamma=['scale'])
# Hyperparameter for RF
param_grid_rf = dict(max_depth=[int(k*np.sqrt(X_all.shape[0]))
                                for k in [0.5, 1, 2]])
print(param_grid_lr)
print(param_grid_svm_rbf)
print(param_grid_rf)


# Run each method on each fold
for kk in range(startfold, endfold):
    file_exists = os.path.exists(
        join(output_path, f'yscore_fold{kk}_seed{seed_num}.csv'))
    if file_exists:
        print("File already exists")
    else:
        print(f'#### kk = {kk} ####')
        print(f'test idx: {te}')

        # Create train/test split
        tr = cv_idx_dict_all[kk]['tr']
        te = cv_idx_dict_all[kk]['te']
        X_tr, y_tr = X_all[tr], y_all[tr]
        X_te, y_te = X_all[te], y_all[te]

        # Baseline
        yscore_bl, yhat_bl = run_bl(X_tr.copy(), y_tr.copy(), X_te.copy(),
                                    task_type, random_state=seed_num)
        print('* Done baseline.')

        # Run SVM-rbf
        yscore_svm_rbf, yhat_svm_rbf = run_svm_rbf(
            X_tr.copy(), y_tr.copy(),
            X_te.copy(),
            task_type, scoring,
            param_grid=param_grid_svm_rbf,
            random_state=seed_num, n_jobs=n_jobs)
        print('* Done SVM-RBF.')

        # Run linear/logistic regression
        yscore_lr_l1, yhat_lr_l1 = run_lr_l1(X_tr.copy(), y_tr.copy(),
                                             X_te.copy(),
                                             task_type, scoring,
                                             param_grid=param_grid_lr,
                                             random_state=seed_num,
                                             n_jobs=n_jobs)
        print('* Done linear/logistic regression with l1 penalty.')

        # Run random forest
        yscore_rf, yhat_rf = run_rf(X_tr.copy(), y_tr.copy(), X_te.copy(),
                                    task_type, scoring,
                                    param_grid=param_grid_rf,
                                    random_state=seed_num, n_jobs=n_jobs)
        print('* Done RF.')

        # Run c-lasso
        yscore_cl, yhat_cl = run_classo(X_tr.copy(), y_tr.copy(), X_te.copy(),
                                        task_type, random_state=seed_num)
        print("* Done c-lasso")

        # Run KernelBiome
        yscore_kb, yhat_kb = run_kb(
            X_tr.copy(), y_tr.copy(), X_te.copy(),
            task_type, scoring,
            param_grid=None, models=None,
            outer_cv_type=None, grp=None,
            n_hyper_grid=20,
            random_state=seed_num, n_jobs=n_jobs,
            outpath=join(
                output_path,
                f'best_kernels_fold{kk}_seed{seed_num}.csv'))
        print("* Done KernelBiome")

        # # Run Aitchison-kernel with zero imputation tuning
        # yscore_ak, yhat_ak = run_kb(X_tr.copy(), y_tr.copy(), X_te.copy(),
        #                             task_type, scoring,
        #                             param_grid=None,
        #                             models=aitchison_models,
        #                             outer_cv_type=None, grp=None,
        #                             n_hyper_grid=20,
        #                             random_state=seed_num, n_jobs=n_jobs,
        #                             outpath=None)
        # print("* Done Aitchison kernel only")

        # Run Aitchison-kernel with default zero imputation
        yscore_ka, yhat_ka = run_kb(X_tr.copy(), y_tr.copy(), X_te.copy(),
                                    task_type, scoring,
                                    param_grid=None,
                                    models=aitchison_models_single,
                                    outer_cv_type=None, grp=None,
                                    n_hyper_grid=20,
                                    random_state=seed_num, n_jobs=n_jobs,
                                    outpath=None)
        print("* Done Aitchison kernel only")

        # Collect results
        yscore_df = pd.DataFrame({
            'Baseline': yscore_bl,
            'SVM-RBF': yscore_svm_rbf,
            'Lin/Log-L1': yscore_lr_l1,
            'RF': yscore_rf,
            'LogCont-L1': yscore_cl,
            'KernelBiome': yscore_kb,
            'KB_Aitchison': yscore_ka
        })
        yhat_df = pd.DataFrame({
            'Baseline': yhat_bl,
            'SVM-RBF': yhat_svm_rbf,
            'Lin/Log-L1': yhat_lr_l1,
            'RF': yhat_rf,
            'LogCont-L1': yhat_cl,
            'KernelBiome': yhat_kb,
            'KB_Aitchison': yhat_ka
        })
        # Save all results
        yscore_df.to_csv(
            join(output_path, f'yscore_fold{kk}_seed{seed_num}.csv'),
            index=False)
        yhat_df.to_csv(
            join(output_path, f'yhat_fold{kk}_seed{seed_num}.csv'),
            index=False)
        print("Results for fold saved!")
