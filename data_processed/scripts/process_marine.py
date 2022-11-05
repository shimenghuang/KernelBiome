from os.path import join
import pandas as pd


# Paths
data_path_in = "data_original/Marine"
data_path_out = "data_processed"


def process_data(data_path_in, data_path_out):

    # reads in data which is p x n
    data_abs = pd.read_csv(join(data_path_in, "marine_X.csv"))
    data_abs.set_index('Unnamed: 0', inplace=True)  # needed for group_taxa
    y_df = pd.read_csv(join(data_path_in, "marine_y.csv"))
    y_df.set_index('Unnamed: 0', inplace=True)
    y_df.rename(columns={'x': 'y'}, inplace=True)

    data_df = data_abs.T.reset_index(drop=True)
    data_df['y'] = y_df['y'].to_numpy()

    data_df.to_csv(join(data_path_out, "marine.csv"), index=False)


process_data(data_path_in, data_path_out)
