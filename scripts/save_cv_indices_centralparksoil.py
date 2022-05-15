import numpy as np
from sklearn.model_selection import KFold
import load_centralparksoil

seed_num = 2022
data_path = "/Users/hrt620/Documents/projects/kernelbiome_proj/kernelbiome_clean/data/CentralParkSoil"
X_df, y, label = load_centralparksoil.main(
    data_path=data_path, seed_num=seed_num)

n_compare = 50
k_fold = KFold(n_compare, shuffle=False)  # Note: do not shuffle
tr_list = []
te_list = []
for kk, (tr, te) in enumerate(k_fold.split(X_df.T, y)):
    print(f'-- kk = {kk} --')
    tr_list.append(tr)
    te_list.append(te)
np.savez(
    f"output/centralparksoil_compare_{n_compare}cv_idx.npz", tr_list=tr_list, te_list=te_list)
