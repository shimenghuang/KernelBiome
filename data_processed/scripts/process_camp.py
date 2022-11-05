from os.path import join
import pandas as pd


data_path_in = "data_original/CAMP"
data_path_out = "data_processed"


def process_data(data_path_in, data_path_out):

    # read in abundance data
    data_abundance = pd.read_csv(join(
        data_path_in,
        "CAMP.16s_DADA2.taxon_abundance.tsv"),
                                 sep="\t",
                                 index_col=0, header=0).fillna(0).T
    data_abundance.index = data_abundance.index.astype(int)

    # read in meta data
    data_meta = pd.read_csv(join(
        data_path_in,
        "CAMP.16s_DADA2.sample_details.tsv"),
                            sep="\t",
                            index_col=0,
                            header=0)
    y_df = pd.DataFrame({
        'y': data_meta['Case or control subject']})
    y_df['y'] = y_df['y'].apply(lambda x: 1 if x == 'Control' else -1)

    # Combine data
    data_df = data_abundance.join(y_df)
    data_df.to_csv(join(data_path_out, "camp.csv"), index=False)


process_data(data_path_in, data_path_out)
