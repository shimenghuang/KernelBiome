from os.path import join
import pandas as pd
import numpy as np

# Paths
data_path_in = "data_original/RMP"
data_path_out = "data_processed"


def process_data(data_path_in, data_path_out):
    """Returns a pandas.DataFrame with rows being the observations and
    first p columns being the features, and last column being 'y', the
    response.  Note: this dataset is only the relative abundance

    """
    # RMP (relative abundance data)
    data_part1 = pd.read_csv(
        join(data_path_in, "RMP_DiseaseCohort_nature24460.tsv"), sep="\t")
    data_part2 = pd.read_csv(
        join(data_path_in, "RMP_DiseaseCohort_66healthy_nature24460.tsv"),
        sep="\t")
    data_df = pd.concat([data_part1, data_part2])
    data_df.set_index('ID', inplace=True)
    data_df['y'] = np.concatenate(
        [np.repeat(1, data_part1.shape[0]),
         np.repeat(-1, data_part2.shape[0])])

    data_df.to_csv(join(data_path_out, "rmp.csv"), index=False)

    # QMP (absolute abundance data)
    data_part1 = pd.read_csv(
        join(data_path_in, "QMP_DiseaseCohort_nature24460.tsv"), sep="\t")
    data_part2 = pd.read_csv(
        join(data_path_in,
             "QMP_DiseaseCohort_66healthy_nature24460.tsv"), sep="\t")
    data_df = pd.concat([data_part1, data_part2])
    data_df.set_index('ID', inplace=True)
    data_df['y'] = np.concatenate(
        [np.repeat(1, data_part1.shape[0]),
         np.repeat(-1, data_part2.shape[0])])

    data_df.to_csv(join(data_path_out, "qmp.csv"), index=False)


process_data(data_path_in, data_path_out)
