# %%
# setup
# ^^^^^^
import sys  # nopep8
from os.path import join  # nopep8

on_computerome = True  # nopep8
fold_idx = int(sys.argv[1])  # nopep8
print(f"fold: {fold_idx}")  # nopep8

if on_computerome:
    sys.path.insert(0, "./kernelbiome/")  # nopep8 # server path
    file_dir = "kernelbiome/notebooks/supervised/"  # server
else:
    sys.path.insert(0, "../../")  # nopep8 # local path
    file_dir = ""  # local

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

exec(open(join(file_dir, "run_cirrhotic_server_prep.py")).read())

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
