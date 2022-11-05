# %%
# libs
# ^^^^^^

# Note: add path so it also works for interactive run in vscode...
import sys  # nopep8
sys.path.insert(0, "../../")  # nopep8

import os
from os.path import join
import pandas as pd
from helpers.process_taxa import (filter_taxa, group_taxa)

# %%
# Paths
# ^^^^^^
# data_path_in = "data_original/MLRepo/qin2014"
# data_path_out = "data_processed"

print(f'cwd: {os.getcwd()}')
if os.getcwd().endswith('data_processed/scripts'):
    data_path_in = "../../data_original/MLRepo/qin2014"
    data_path_out = "../../data_processed"
else:
    # assuming running from kernelbiome_tmp
    data_path_in = "data_original/MLRepo/qin2014"
    data_path_out = "data_processed"


def process_data(data_path_in, data_path_out):

    # reads in data which is p x n
    data_abs = pd.read_csv(join(data_path_in, "taxatable.txt"), sep="\t")
    data_abs.set_index('#OTU ID', inplace=True)  # needed for group_taxa
    y_df = pd.read_csv(
        join(data_path_in, "task-healthy-cirrhosis.txt"), sep="\t")
    y_df.set_index('#SampleID', inplace=True)
    y_df.rename(columns={'Var': 'y'}, inplace=True)
    y_df['y'] = y_df['y'].apply(lambda x: 1 if x == 'Healthy' else -1)

    data_abs = group_taxa(data_abs, grp_to="species")
    data_abs = filter_taxa(data_abs, [0.25, 5], verbose=True)
    data_df = data_abs.T.join(y_df)

    data_df.to_csv(join(data_path_out, "cirrhosis.csv"), index=False)


process_data(data_path_in, data_path_out)

# %%
