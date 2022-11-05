from os.path import join
import pandas as pd

# Paths
data_path_in = "data_original/GenomeMed"
data_path_out = "data_processed"


def process_data(data_path_in, data_path_out):

    data_orig = pd.read_csv(join(
        data_path_in,
        "glne007.final.an.unique_list.0.03.subsample.0.03.filter.shared"),
                            sep="\t")
    data_met = pd.read_csv(join(data_path_in, "metadata.tsv"), sep="\t")

    data_abs = data_orig.copy().drop(
        ['label', 'numOtus', 'Unnamed: 338'], axis=1)
    data_abs.set_index('Group', inplace=True)
    y_df = data_met.copy()[['sample', 'dx']]
    y_df.rename(columns={'dx': 'y'}, inplace=True)
    y_df.set_index('sample', inplace=True)
    # y_df['y'] = y_df['y'].apply(lambda x: -1 if x == 'normal' else 1)
    # genomemed_tumor
    y_df['y'] = y_df['y'].apply(
        lambda x: 1 if x == 'cancer' else -1)  # genomemed_cancer
    data_df = data_abs.join(y_df)

    data_df.to_csv(join(data_path_out, "genomemed_cancer.csv"), index=False)


process_data(data_path_in, data_path_out)
