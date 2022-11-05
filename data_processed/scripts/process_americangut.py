from os.path import join
import pandas as pd
from helpers.process_taxa import filter_taxa

# Paths
data_path_in = "data_original/AmericanGut"
data_path_out = "data_processed"


def process_data(data_path_in, data_path_out):

    # UK sample
    # reads in data which is p x n
    data_abs = pd.read_csv(join(data_path_in, "ag_X_uk.csv"))
    data_abs.set_index('Unnamed: 0', inplace=True)  # needed for group_taxa
    y_df = pd.read_csv(join(data_path_in, "ag_y_uk.csv"))
    y_df.set_index('Unnamed: 0', inplace=True)
    y_df.rename(columns={'x': 'y'}, inplace=True)

    # Filter taxa with prevalence >= 0.25 and median_count >=5
    data_abs = filter_taxa(data_abs, [0.25, 5], verbose=True)
    data_df = data_abs.T.reset_index(drop=True)
    data_df['y'] = y_df['y'].to_numpy()

    data_df.to_csv(join(data_path_out, "americangut_uk.csv"), index=False)

    # # USA sample
    # # reads in data which is p x n
    # data_abs = pd.read_csv(join(data_path_in, "ag_X_usa.csv"))
    # data_abs.set_index('Unnamed: 0', inplace=True)  # needed for group_taxa
    # y_df = pd.read_csv(join(data_path_in, "ag_y_usa.csv"))
    # y_df.set_index('Unnamed: 0', inplace=True)
    # y_df.rename(columns={'x': 'y'}, inplace=True)

    # data_abs = filter_taxa(data_abs, [0.25, 5], verbose=True)
    # data_df = data_abs.T.reset_index(drop=True)
    # data_df['y'] = y_df['y'].to_numpy()

    # data_df.to_csv(join(data_path_out, "americangut_usa.csv"), index=False)


process_data(data_path_in, data_path_out)
