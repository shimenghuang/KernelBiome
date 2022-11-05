###
# Libraries
###

# Note: add path so it also works for interactive run in vscode...
import sys  # nopep8
sys.path.insert(0, "../../")  # nopep8

import os
from os.path import join
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from helpers.load_data import load_processed
from matplotlib import rc


width = 6.34
fontsize = 6
rc('font', **{'size': fontsize})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')

# Color codes
ccodes = mcolors.TABLEAU_COLORS


print(f'cwd: {os.getcwd()}')
if os.getcwd().endswith('post_analysis'):
    data_path = "../../data_processed"
    results_path = "../../experiments/post_analysis/results/"
    exp_path = "../../experiments/post_analysis/"
else:
    # assuming running from kernelbiome_tmp
    data_path = "data_processed"
    results_path = "experiments/post_analysis/results/"
    exp_path = "experiments/post_analysis/"


###
# Load cirrhosis data and preprocess
###

data_name = "cirrhosis"
X, y, label_all, group_all = load_processed(data_name, data_path)
disease_label = np.array(['Cirrhosis']*len(y))
disease_label[y == 1] = 'Healthy'
X /= X.sum(axis=1)[:, None]
print(X.shape)

# Screen data
cfis_screen = np.load(os.path.join(
    results_path, "CFIscreen_cirrhosis.npy"))
ind = np.argsort(np.abs(cfis_screen))[-50:]
X = X[:, ind]
X /= X.sum(axis=1)[:, None]
label_all = label_all[ind]
print(X.shape)

###
# shorten the labels
###


def shorten_label(lbl_orig):
    lbl_split = lbl_orig.split(';')
    return lbl_split[-1][3:]


label_short = np.array([shorten_label(lbl) for lbl in label_all])


# ###
# # Standard PCA-based analysis with CLR-transform
# ###

# # CLR-transform X
# minX = X[X != 0].min()
# Xclr = X.copy() + minX/2
# Xclr = np.log(Xclr) - np.log(Xclr).mean(axis=1)[:, None]

# # PCA
# pca = PCA(2)
# pca.fit(Xclr)
# pcs = pca.transform(Xclr)

# # Generate plot
# df = pd.DataFrame({'x1': pcs[:, 0],
#                    'x2': pcs[:, 1],
#                    'disease': disease_label})
# groups = df.groupby('disease')
# markers = {'Healthy': '^',
#            'Cirrhosis': 'o'}

# colors = ['tab:purple', 'tab:green']
# fig, ax = plt.subplots(1, 1)
# for ii, (name, group) in enumerate(groups):
#     ax.plot(group.x1, group.x2,
#             marker=markers[str(name)],
#             linestyle='',
#             markersize=3,
#             color=ccodes[colors[ii]],
#             label=str(name))
# ax.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)
# ax.legend(bbox_to_anchor=(0.25, 0.95, 0.5, .1),
#           loc='upper center', handletextpad=0.2,
#           ncol=2, mode="expand", borderaxespad=-0.2)
# ax.set_xlabel("component 1")
# ax.set_ylabel("component 2")
# fig.set_size_inches(3, 3)
# for spine in ax.spines.values():
#     spine.set(color='gray', linewidth=1, alpha=0.2)
# fig.savefig(join(exp_path, "clr-pca_cirrhosis.pdf"), bbox_inches='tight')
# plt.show()


###
# Summary statistics vs diversity plot
###

df_diversity = pd.read_csv(
    os.path.join(results_path, 'diversity_cirrhosis.csv'))

groups = df_diversity.groupby('disease')
plot_dict1 = []
plot_dict2 = []
names = []
for name, group in groups:
    names.append(name)
    plot_dict1.append(group['kerneldist'])
    plot_dict2.append(group['simpson-diversity'])

colors = ['tab:blue', 'tab:gray']
fig, ax = plt.subplots(1, 1)
p1 = ax.boxplot(plot_dict1, positions=[1-0.15, 1+0.15], notch=True,
                widths=0.2, patch_artist=True,
                boxprops=dict(
                facecolor=ccodes[colors[0]],
                edgecolor='white'),
                flierprops=dict(markersize=2))
ax.set_ylabel('Kernel health score', labelpad=1, color=ccodes[colors[0]])

axs2 = ax.twinx()
p2 = axs2.boxplot(plot_dict2, positions=[2-0.15, 2+0.15], notch=True,
                  widths=0.2, patch_artist=True,
                  boxprops=dict(
                  facecolor=ccodes[colors[1]],
                  edgecolor='white'),
                  flierprops=dict(markersize=2))
axs2.set_ylabel('Simpson diversity', labelpad=7,
                color=ccodes[colors[1]], rotation=270)
p1['medians'][0].set_color('black')
p1['medians'][1].set_color('black')
p2['medians'][0].set_color('black')
p2['medians'][1].set_color('black')

ax.set_xticks([1-0.15, 1+0.15, 2-0.15, 2+0.15])
ax.set_xticklabels(['Cirrhosis', 'Healthy', 'Cirrhosis', 'Healthy'],
                   rotation=60, ha="right")
for spine in ax.spines.values():
    spine.set(color='gray', linewidth=1, alpha=0.2)
for spine in axs2.spines.values():
    spine.set_visible(False)

# adjustments to ticks
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useOffset=False)
ax.tick_params(axis='y', left=False, pad=1)
ax.tick_params(axis='x', bottom=False, pad=1)
axs2.tick_params(axis='y', right=False, pad=1)
ax.tick_params(axis='y', labelsize=fontsize*0.65)
axs2.tick_params(axis='y', labelsize=fontsize*0.65)
ax.yaxis.offsetText.set_fontsize(fontsize*0.65)

fig.set_size_inches(width/7, width/4)
fig.savefig(join(exp_path, "sum_stat_and_sim_div_cirrhosis.pdf"),
            bbox_inches='tight')


###
# Kernel PCA
###

def make_legend_hdl(label, marker='s', color=ccodes['tab:red'], alpha=1):
    return mlines.Line2D([], [], color=color,
                         marker=marker, linestyle='None', alpha=alpha,
                         markersize=1, label=label)


df = pd.read_csv(
    os.path.join(results_path, 'kernelPCA_cirrhosis.csv'))
cc = np.load(os.path.join(results_path, 'kernelPCA_cirrhosis.npy'))

fig = plt.figure()
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2, fig=fig)
groups = df.groupby('disease')
markers = {'Healthy': '^',
           'Cirrhosis': 'o'}
colors = ['tab:purple', 'tab:green']
for ii, (name, group) in enumerate(groups):
    ax1.plot(group.x1, group.x2,
             marker=markers[str(name)],
             color=ccodes[colors[ii]],
             linestyle='',
             markersize=3,
             label=str(name))
ax1.set_xlabel("component 1", labelpad=1)
ax1.set_ylabel("component 2", labelpad=0.2)
ax1.tick_params(pad=1, axis='x')
ax1.tick_params(pad=1, axis='y')
ax1.legend(bbox_to_anchor=(0.25, 0.95, 0.5, .1), loc='upper center',
           handletextpad=0.2, ncol=2, mode="expand",
           borderaxespad=-0.2, fancybox=False)
for spine in ax1.spines.values():
    spine.set(color='gray', linewidth=1, alpha=0.2)
# font sizes of tick values
ax1.tick_params(axis='y', labelsize=fontsize*0.65)
ax1.tick_params(axis='x', labelsize=fontsize*0.65)

num_top = 3
# Contribution component 1
ax2 = plt.subplot2grid((2, 2), (0, 1), fig=fig)
comp1_index = np.argsort(cc[:, 0])
comp1_sort = cc[comp1_index, 0]
ax2.bar(np.arange(num_top, cc.shape[0]-num_top),
        comp1_sort[num_top:-num_top], alpha=0.49)
for k in range(num_top):
    ax2.bar(k, comp1_sort[k], color=ccodes['tab:red'], alpha=0.7**k)
    ax2.bar(len(comp1_sort)-1-k, comp1_sort[len(comp1_sort)-1-k],
            color=ccodes['tab:green'], alpha=0.7**k)
labels_top1 = label_short[comp1_index[-3:]]
labels_bot1 = label_short[comp1_index[:3]]
# add legend
l1 = make_legend_hdl(labels_top1[0], color=ccodes['tab:green'], alpha=1)
l2 = make_legend_hdl(labels_top1[1], color=ccodes['tab:green'], alpha=0.7)
l3 = make_legend_hdl(labels_top1[2], color=ccodes['tab:green'], alpha=0.49)
l4 = make_legend_hdl(labels_bot1[0], color=ccodes['tab:red'], alpha=1)
l5 = make_legend_hdl(labels_bot1[1], color=ccodes['tab:red'], alpha=0.7)
l6 = make_legend_hdl(labels_bot1[2], color=ccodes['tab:red'], alpha=0.49)
lgd1 = plt.legend(handles=[l1, l2, l3],
                  loc='upper left',
                  labelspacing=0.2,
                  columnspacing=0.5,
                  bbox_to_anchor=(-0.02, 1.0),
                  handletextpad=-0.5,  # space between marker and text
                  fancybox=False,
                  shadow=False, ncol=1,
                  frameon=False)
lgd2 = plt.legend(handles=[l4, l5, l6],
                  loc='lower right',
                  labelspacing=0.2,
                  columnspacing=0.5,
                  bbox_to_anchor=(1, 0),
                  handletextpad=-0.5,  # space between marker and text
                  fancybox=False,
                  shadow=False, ncol=1,
                  frameon=False)
ax2.add_artist(lgd1)
ax2.add_artist(lgd2)
ax2.yaxis.tick_right()
ax2.set_title("component 1", pad=2.0)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useOffset=False)
for spine in ax2.spines.values():
    spine.set(color='gray', linewidth=1, alpha=0.2)
# font sizes of tick values
ax2.tick_params(axis='y', labelsize=fontsize*0.65)
ax2.tick_params(axis='x', labelsize=fontsize*0.65)
ax2.yaxis.offsetText.set_fontsize(fontsize*0.65)

# Contribution component 2
ax3 = plt.subplot2grid((2, 2), (1, 1), fig=fig)
comp2_index = np.argsort(cc[:, 1])
comp2_sort = cc[comp2_index, 1]
ax3.bar(np.arange(num_top, cc.shape[0]-num_top),
        comp2_sort[num_top:-num_top], alpha=0.49)
for k in range(num_top):
    ax3.bar(k, comp2_sort[k], color=ccodes['tab:red'], alpha=0.7**k)
    ax3.bar(len(comp2_sort)-1-k, comp2_sort[len(comp2_sort)-1-k],
            color=ccodes['tab:green'], alpha=0.7**k)
labels_top2 = label_short[comp2_index[-3:]]
labels_bot2 = label_short[comp2_index[:3]]
# add legend
l1 = make_legend_hdl(labels_top2[0], color=ccodes['tab:green'], alpha=1)
l2 = make_legend_hdl(labels_top2[1], color=ccodes['tab:green'], alpha=0.7)
l3 = make_legend_hdl(labels_top2[2], color=ccodes['tab:green'], alpha=0.49)
l4 = make_legend_hdl(labels_bot2[0], color=ccodes['tab:red'], alpha=1)
l5 = make_legend_hdl(labels_bot2[1], color=ccodes['tab:red'], alpha=0.7)
l6 = make_legend_hdl(labels_bot2[2], color=ccodes['tab:red'], alpha=0.49)
lgd1 = plt.legend(handles=[l1, l2, l3],
                  loc='upper left',
                  labelspacing=0.2,
                  columnspacing=0.5,
                  bbox_to_anchor=(-0.02, 1.0),
                  handletextpad=-0.5,  # space between marker and text
                  fancybox=False,
                  shadow=False, ncol=1,
                  frameon=False)
lgd2 = plt.legend(handles=[l4, l5, l6],
                  loc='lower right',
                  labelspacing=0.2,
                  columnspacing=0.5,
                  bbox_to_anchor=(1, 0),
                  handletextpad=-0.5,  # space between marker and text
                  fancybox=False,
                  shadow=False, ncol=1,
                  frameon=False)
ax3.add_artist(lgd1)
ax3.add_artist(lgd2)
ax3.yaxis.tick_right()
ax3.set_xlabel("sorted species")
ax3.set_title("component 2", pad=2.0)
ax3.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useOffset=False)
for spine in ax3.spines.values():
    spine.set(color='gray', linewidth=1, alpha=0.2)
# font sizes of tick values
ax3.tick_params(axis='y', labelsize=fontsize*0.65)
ax3.tick_params(axis='x', labelsize=fontsize*0.65)
ax3.yaxis.offsetText.set_fontsize(fontsize*0.65)

# Finalize figure
fig.tight_layout(pad=0.5)
fig.subplots_adjust(wspace=0.05, hspace=0.2)
fig.set_size_inches(width, width/2)
fig.text(1.0, 0.5, 'contribution to response',
         va='center', rotation=270)
plt.savefig(join(exp_path, "kpca_cirrhosis.pdf"),
            bbox_inches='tight')


###
# kPCA but without the loadings
###

df = pd.read_csv(
    os.path.join(results_path, 'kernelPCA_cirrhosis.csv'))
cc = np.load(os.path.join(results_path, 'kernelPCA_cirrhosis.npy'))

fig, ax = plt.subplots(1, 1)
groups = df.groupby('disease')
markers = {'Healthy': '^',
           'Cirrhosis': 'o'}
colors = ['tab:purple', 'tab:green']
for ii, (name, group) in enumerate(groups):
    ax.plot(group.x1, group.x2,
            marker=markers[str(name)],
            color=ccodes[colors[ii]],
            linestyle='',
            markersize=1,
            label=str(name))
ax.set_xlabel("component 1", labelpad=1)
ax.set_ylabel("component 2", labelpad=0.5)
ax.legend(bbox_to_anchor=(0.1, 1, 0.8, .1),
          loc='upper center', handletextpad=0.0,
          ncol=2, mode="expand", borderaxespad=-0.2,
          frameon=False)
for spine in ax.spines.values():
    spine.set(color='gray', linewidth=1, alpha=0.2)
# adjustments for ticks
ax.tick_params(axis='y', labelsize=fontsize*0.65)
ax.tick_params(axis='x', labelsize=fontsize*0.65)
ax.tick_params(axis='x', bottom=False, pad=0)
ax.tick_params(axis='y', left=False, pad=0)

plt.tight_layout()
fig.set_size_inches(width/4, width/4)
plt.savefig(join(exp_path, "kpca_only_cirrhosis.pdf"),
            bbox_inches='tight')


###
# CFI plot
###

cfis = np.load(os.path.join(results_path, "CFI_cirrhosis.npy"))

num_top = 3
cfi_sort_ind = np.argsort(cfis)
cfis_sorted = cfis[cfi_sort_ind]
fig, ax = plt.subplots(1, 1)
ax.bar(np.arange(num_top, len(cfis)-num_top), cfis_sorted[num_top:-num_top],
       color=ccodes['tab:blue'], alpha=0.49)
handles_top = []
handles_bot = []
for k in range(num_top):
    l1 = label_short[cfi_sort_ind[k]]
    ax.bar(k, cfis_sorted[k], color=ccodes['tab:red'], alpha=0.7**k, label=l1)
    handles_top.append(make_legend_hdl(
        l1, color=ccodes['tab:red'], alpha=0.7**k))
for k in range(num_top):
    l2 = label_short[cfi_sort_ind[len(cfis)-1-k]]
    ax.bar(len(cfis)-1-k, cfis_sorted[len(cfis)-1-k],
           color=ccodes['tab:green'], alpha=0.7**k, label=l2)
    handles_bot.append(make_legend_hdl(
        l2, color=ccodes['tab:green'], alpha=0.7**k))
for spine in ax.spines.values():
    spine.set(color='gray', linewidth=1, alpha=0.2)
lgd1 = plt.legend(handles=handles_top,
                  loc='upper left',
                  labelspacing=0.2,
                  columnspacing=0.5,
                  bbox_to_anchor=(-0.02, 1.0),
                  handletextpad=-0.5,  # space between marker and text
                  fancybox=False,
                  shadow=False, ncol=1,
                  frameon=False,
                  fontsize=fontsize*0.7)
lgd2 = plt.legend(handles=handles_bot,
                  loc='lower right',
                  labelspacing=0.2,
                  columnspacing=0.5,
                  bbox_to_anchor=(1, 0),
                  handletextpad=-0.5,  # space between marker and text
                  fancybox=False,
                  shadow=False, ncol=1,
                  frameon=False,
                  fontsize=fontsize*0.7)
ax.add_artist(lgd1)
ax.add_artist(lgd2)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useOffset=False)
ax.tick_params(axis='y', labelsize=fontsize*0.65)
ax.tick_params(axis='x', labelsize=fontsize*0.65)
ax.yaxis.offsetText.set_fontsize(fontsize*0.65)
ax.tick_params(axis='x', bottom=False, labelbottom=False)
ax.tick_params(axis='y', left=False, pad=0)
ax.set_ylabel('CFI', labelpad=0.1)
ax.set_xlabel("sorted species", labelpad=2.0)
plt.tight_layout()
fig.set_size_inches(width/4, width/8)
plt.savefig(join(exp_path, "cfi_cirrhosis.pdf"),
            bbox_inches='tight')
