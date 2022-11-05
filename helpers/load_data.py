import pandas as pd
from os.path import join


def load_processed(data_name, data_path, reload=False):
    if reload:
        print("rerun data")
    data_df = pd.read_csv(join(data_path, f'{data_name}.csv'))
    y = data_df['y'].to_numpy()
    if 'grp' in data_df.columns:
        grp = data_df['grp'].to_numpy()
        X_df = data_df.drop(['y', 'grp'])
    else:
        grp = None
        X_df = data_df.drop('y', axis=1)
    label = X_df.columns.to_numpy()
    return X_df.to_numpy().astype('float'), y, label, grp
