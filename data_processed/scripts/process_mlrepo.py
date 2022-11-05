# %%
# libs
# ^^^^^^
from os.path import join
import pandas as pd
import numpy as np
import re
import os

# %%
# paths
# ^^^^^^

# data_path_in = "data_original/MLRepo"
# data_path_out = "data_processed"

print(f'cwd: {os.getcwd()}')
if os.getcwd().endswith('data_processed/scripts'):
    data_path_in = "../../data_original/MLRepo"
    data_path_out = "../"
else:
    # assuming running from kernelbiome_tmp
    data_path_in = "data_original/MLRepo"
    data_path_out = "data_processed"


# %%
# create datasets
# ^^^^^^

def process_data(data_path_in, data_path_out):

    # get folders
    folders = os.listdir(data_path_in)
    folders = [f for f in folders if not f.startswith('.')]

    # Iterate over all folders
    for folder in folders:
        curr_path = join(data_path_in, folder)
        print(f'Working on folder {folder}')
        # read taxa
        if os.path.exists(join(curr_path, 'gg/taxatable.txt')):
            path_tmp = join(curr_path, 'gg/taxatable.txt')
        else:
            path_tmp = join(curr_path, 'taxatable.txt')
        taxa_data = pd.read_csv(path_tmp,
                                header=0, index_col=0,
                                sep='\t').fillna(0).T
        taxa_data.index = taxa_data.index.astype(str)
        # read task
        all_files = os.listdir(curr_path)
        task_files = [ff for ff in all_files if "task" in ff]
        for tf in task_files:
            task_data = pd.read_csv(join(curr_path, tf),
                                    header=0, index_col=0,
                                    sep='\t')
            varkey = 'Var'
            task_data = task_data[[varkey]].dropna()
            if not pd.api.types.is_numeric_dtype(task_data[varkey]):
                task_data[varkey] = task_data[varkey].astype('category')
                task_data[varkey] = task_data[varkey].cat.codes * 2 - 1
                reg_type = "classification"
            else:
                reg_type = "regression"
            task_data.rename(columns={varkey: 'y'}, inplace=True)
            task_data.index = task_data.index.astype(str)
            data_full = taxa_data.join(task_data, how='inner')
            # Save file
            if reg_type == "classification":
                counts = np.unique(data_full['y'], return_counts=True)[1]
                large_enough = np.sum(counts >= 15) == 2
            else:
                large_enough = data_full.shape[0] >= 30
            if large_enough:
                if len(task_files) == 1:
                    name = 'mlr_' + folder
                else:
                    tn = re.search('task-(.+?).txt', tf).group(1)
                    name = 'mlr_' + folder + '_' + tn
                print(f'Saving {name} with size {data_full.shape}')
                data_full.to_csv(join(data_path_out, name + ".csv"),
                                 index=False)


process_data(data_path_in, data_path_out)

# %%
