# %%
# load libs
# ^^^^^^

on_computerome = False   # nopep8
import sys  # nopep8
if on_computerome:
    sys.path.insert(0, "./KernelBiome/")  # nopep8
else:
    sys.path.insert(0, "../")  # nopep8

# from os.path import join
import matplotlib.pyplot as plt
from src.helpers_weighting import *
import load_cirrhotic

# %%
# load data
# ^^^^^^

if on_computerome:
    file_dir = "kernelbiome_clean/scripts"  # dir of scripts being called
    data_path = "kernelbiome_clean/data/MLRepo/qin2014"  # path to load data
else:
    file_dir = ""  # local
    data_path = "/Users/hrt620/Documents/projects/kernelbiome_proj/kernelbiome_clean/data/MLRepo/qin2014"

# exec(open(join(file_dir, "load_cirrhotic.py")).read())
X_df, label = load_cirrhotic.main(data_path=data_path, seed_num=2022)

# %%
# compute unifrac weight
# ^^^^^^

# name of biom and tree file
biom_file = "output/qin2014_1sample.biom"
tree_file = "output/qin2014_tree.tre"

levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
delim = ";"
w_unifrac = compute_unifrac_similarity(
    label, biom_file, tree_file, levels=levels, delim=delim, ensure_psd=True)

# W^{unifrac} with M^A
res = compute_unifrac(label, biom_file, tree_file,
                      set_branch_length=True,
                      branch_length=1.0,
                      levels=levels,
                      delim=delim,
                      table_id="Dataset")
dist_unifrac = res["UnifracTable"].data
M = 1.0-dist_unifrac
D = np.diag(1.0/np.sqrt(M.sum(axis=1)))
W = D.dot(M).dot(D)
np.save("output/cirrhotic_w_unifrac_MA.npy", W)
plt.imshow(W)
plt.xlabel("Species")
plt.ylabel("Species")
plt.savefig("output/cirrhotic_w_unifrac_MA.pdf")

# W^{unifrac} with M^B
M = dist_unifrac.T.dot(dist_unifrac)
D = np.diag(1.0/np.sqrt(M.sum(axis=1)))
W = D.dot(M).dot(D)
np.save("output/cirrhotic_w_unifrac_MB.npy", W)
plt.imshow(W)
plt.xlabel("Species")
plt.ylabel("Species")
plt.savefig("output/cirrhotic_w_unifrac_MB.pdf")

# %%
