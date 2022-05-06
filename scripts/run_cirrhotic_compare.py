# %%
# setup file paths
# ^^^^^^
import sys  # nopep8
from os.path import join
from classo import classo_problem
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

on_computerome = True  # nopep8
fold_idx = int(sys.argv[1])  # nopep8
print(f"fold: {fold_idx}")  # nopep8


# %%
# set up paths
# ^^^^^^
if on_computerome:
    data_path = "kernelbiome_clean/data/MLRepo/qin2014"  # path to load data
else:
    data_path = "/Users/hrt620/Documents/projects/kernelbiome_proj/kernelbiome_clean/data/MLRepo/qin2014"


# output file
output_path = "."
do_save = True
do_save_file = join(
    output_path, f"res_kb_cirrhotic_fold_{fold_idx}.pickle")
do_save_file_weighted_orig = join(
    output_path, f"res_weighted_kb_cirrhotic_orig_fold_{fold_idx}.pickle")
do_save_file_weighted_psd = join(
    output_path, f"res_weighted_kb_cirrhotic_psd_fold_{fold_idx}.pickle")

# print(sys.path)
# print("\n")
# import os
# print(os.listdir("./kernelbiome/"))

# %%
# call prep script
# ^^^^^^

exec(open(join(file_dir, "load_cirrhotic.py")).read())

# data to be used for classo (with added pseudo count)
pseudo_count = 1  # Note: pseudo count seems to matter
X = np.log(pseudo_count + X_df.to_numpy().astype('float').T)

# %%
# median heruisic for RBF and Aitchison-RBF
# ^^^^^^

X_comp = X_df.to_numpy().astype('float').T
X_comp /= X_comp.sum(axis=1)[:, None]

# for rbf
K = squared_euclidean_distances(X_comp, X_comp)
k_triu = K[np.triu_indices(n=K.shape[0], k=1, m=K.shape[1])]
g1 = 1.0/np.median(k_triu)
# g2 = 1.0/np.median(np.sqrt(k_triu))**2
print(g1)
# print(g2)

# for aitchison-rbf
Xc = X_comp + 1e-5
gm_Xc = gmean(Xc, axis=1)
clr_Xc = np.log(Xc/gm_Xc[:, None])
K = squared_euclidean_distances(clr_Xc, clr_Xc)
k_triu = K[np.triu_indices(n=K.shape[0], k=1, m=K.shape[1])]
g2 = 1.0/np.median(k_triu)
print(g2)

# %%
# setup hyperparameter search grid
# ^^^^^^

param_grid_lr = dict(C=[10**x for x in [-3, -2, -1, 0, 1, 2, 3]])
print(param_grid_lr)
param_grid_svm = dict(C=[10**x for x in [-3, -2, -1, 0, 1, 2, 3]])
print(param_grid_svm)
param_grid_kr = dict(alpha=list(np.logspace(-7, 1, 5, base=2)))
print(param_grid_kr)
param_grid_rf = dict(n_estimators=[10, 20, 50, 100, 250, 500])
print(param_grid_rf)
param_grid_baseline = dict(strategy=["mean", "median"])
print(param_grid_baseline)

# %%
# setup for KB estimator
# ^^^^^^

kernel_params_dict = default_kernel_params_grid()
kmat_with_params = get_kmat_with_params(kernel_params_dict)
mod_with_params = kmat_with_params

# %%
# setup for WKB estimator
# ^^^^^^

w_unifrac_ma = np.load(
    join(file_dir, "cirrhotic_w_unifrac_MA.npy"), allow_pickle=True)
w_unifrac_mb = np.load(
    join(file_dir, "cirrhotic_w_unifrac_MB.npy"), allow_pickle=True)

# original unifrac weights
weighted_kernel_params_dict_ma = default_weighted_kernel_params_grid(
    w_unifrac_ma, g1, g2)
weighted_kmat_with_params_orig = get_weighted_kmat_with_params(
    weighted_kernel_params_dict_ma, w_unifrac=w_unifrac_ma)
weighted_mod_with_params_orig = weighted_kmat_with_params_orig

# psd-ed unifrac weights
weighted_kernel_params_dict_mb = default_weighted_kernel_params_grid(
    w_unifrac_mb, g1, g2)
weighted_kmat_with_params_psd = get_weighted_kmat_with_params(
    weighted_kernel_params_dict_mb, w_unifrac=w_unifrac_mb)
weighted_mod_with_params_psd = weighted_kmat_with_params_psd

# %%
# save the CV indices once, later just load
# ^^^^^^

# Save indices for the server:
# import sys  # nopep8
# from os.path import join  # nopep8
# on_computerome = False # for this dataset the data_path matters
# sys.path.insert(0, "../../")  # nopep8 # local path
# file_dir = ""  # local
# exec(open(join(file_dir, "run_cirrhotic_server_prep.py")).read())
# n_compare = 50
# k_fold = KFold(n_compare, shuffle=False)  # Note: do not shuffle
# tr_list = []
# te_list = []
# for kk, (tr, te) in enumerate(k_fold.split(X_df.T, y)):
#     print(f'-- kk = {kk} --')
#     tr_list.append(tr)
#     te_list.append(te)
# np.savez(f"comparison_{n_compare}cv_idx_cirrhotic.npz", tr_list=tr_list, te_list=te_list)

n_compare = 50
comparison_cv_idx = np.load(
    join(file_dir, f"comparison_{n_compare}cv_idx_cirrhotic.npz"), allow_pickle=True)

# %%
# run a comparison CV fold
# ^^^^^^

tr = comparison_cv_idx['tr_list'][fold_idx]
te = comparison_cv_idx['te_list'][fold_idx]

exec(open(join(file_dir, "run_cirrhotic_server_main_part1.py")).read())
exec(open(join(file_dir, "run_cirrhotic_server_main_part2.py")).read())

scores_series = pd.Series({
    'baseline': test_score_baseline,
    'SVC(RBF)': test_score_svc,
    'Logistic': test_score_lr,
    'classo': test_score_classo,
    'RF': test_score_rf,
    'KernelBiome': test_score_kb,
    'OrigUnifracKB': test_score_wkb_orig,
    'PSDUnifracKB': test_score_wkb_psd
})

scores_series.to_csv(
    f"cirrhotic_classo_{pseudo_count}_prescreen_manual_beta0_{beta0}_beta1_{beta1}_fold_{fold_idx}.csv", index=False)

# %%
