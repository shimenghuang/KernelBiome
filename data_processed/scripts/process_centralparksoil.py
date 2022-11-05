# %%
# libs
# ^^^^^^

# Note: add path so it also works for interactive run in vscode...
import sys  # nopep8
sys.path.insert(0, "../../")  # nopep8

import os
from os.path import join
import pandas as pd
from helpers.process_taxa import filter_taxa


# %% Paths
# data_path_in = "data_original/CentralParkSoil"
# data_path_out = "data_processed"

print(f'cwd: {os.getcwd()}')
if os.getcwd().endswith('data_processed/scripts'):
    data_path_in = "../../data_original/CentralParkSoil"
    data_path_out = "../../data_processed"
else:
    # assuming running from kernelbiome_tmp
    data_path_in = "data_original/CentralParkSoil"
    data_path_out = "data_processed"


def process_data(data_path_in, data_path_out):

    # reads in data which is p x n
    data_abs = pd.read_csv(join(data_path_in, "soil_X.csv"))
    data_abs.set_index('Unnamed: 0', inplace=True)  # needed for group_taxa
    y_df = pd.read_csv(join(data_path_in, "soil_y.csv"))
    y_df.set_index('Unnamed: 0', inplace=True)
    y_df.rename(columns={'x': 'y'}, inplace=True)

    data_abs = filter_taxa(data_abs, [0.25, 5], verbose=True)
    data_df = data_abs.T.join(y_df)

    data_df.to_csv(join(data_path_out, "centralparksoil.csv"), index=False)


process_data(data_path_in, data_path_out)

# %%
