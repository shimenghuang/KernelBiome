from os.path import join
import pandas as pd


# Paths
data_path_in = "data_original/Mouse"
data_path_out = "data_processed"


def process_data(data_path_in, data_path_out):

    # read in abundance data
    data_abundance = pd.read_csv(join(
        data_path_in,
        "Relative_Abundance_Table.csv"),
                                 index_col=0, header=0)
    data_meta = data_abundance[['Diet', 'Site', 'Day', 'mouse', 'Cage']]
    data_abundance.drop(['Diet', 'Site', 'Day', 'mouse', 'Cage'],
                        axis=1, inplace=True)

    # process meta data
    y_df = pd.DataFrame({
        'y': data_meta['Diet']})
    y_df['y'] = y_df['y'].apply(lambda x: 1 if x == 'Control' else -1)

    # Save prediction data
    data_df = data_abundance.join(y_df)
    data_df.to_csv(join(data_path_out, "mouse.csv"), index=False)

    # Save full data with meta data
    data_full = data_df.join(data_meta)
    data_full.to_csv(join(data_path_out, "mouse_meta.csv"), index=False)


process_data(data_path_in, data_path_out)
