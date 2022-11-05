##
# Libraries
##

# Note: add path so it also works for interactive run in vscode...
import sys  # nopep8
sys.path.insert(0, "../../")  # nopep8

import os
from os.path import join
import numpy as np
import pandas as pd
from helpers.load_data import load_processed
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib import rc, rcParams


##
# Specify plotting parameters
##

width = 3.17
fontsize = 6
rc('font', **{'size': fontsize})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')

# Color codes
# cnames in the desired order for the methods:
# Baseline, SVM-RBF, Lin/Log-L1, RF, LogCont-L1, KernelBiome, KB_Aitchison
cnames = ['tab:blue', 'tab:brown', 'tab:purple', 'tab:pink', 'tab:cyan',
          'tab:green', 'tab:olive']
ccodes = mcolors.TABLEAU_COLORS


##
# Specify datasets and paths
##

print(f'cwd: {os.getcwd()}')
if os.getcwd().endswith('prediction'):
    data_path = "../../data_processed/"
    exp_path = "../../experiments/prediction/"
else:
    # assuming running from kernelbiome_tmp
    data_path = "data_processed/"
    exp_path = "experiments/prediction/"


# Self-currated datasets (for main text)
seed_nums = list(range(100, 120))
run_name_root = 'banana'
grid_dim_x = 2
grid_dim_y = 4
grid_dim_x_roc = 1
grid_dim_y_roc = 4
datasets = ['rmp', 'camp', 'cirrhosis', 'genomemed_cancer',
            'centralparksoil', 'americangut_uk', 'hiv', 'tara']

##
# helpers to create summary
##


def return_rank(yhat):
    """
    This function converts scores to ranks and breaks ties randomly
    """
    yhat2 = np.zeros(len(yhat))
    sorted_y = np.sort(yhat)
    kk = 0
    for r in np.unique(sorted_y):
        ind = np.where(yhat == r)[0]
        np.random.shuffle(ind)
        for j in ind:
            yhat2[j] = kk
            kk += 1
    return(yhat2)


def gather_results(data_name, data_path, exp_path, run_name_root, seed_nums,
                   cv_type='kfold', n_fold=10):

    yscores = {}
    yhats = {}
    best_kernels = {}

    # Load processed data
    X_all, y_all, label_all, group_all = load_processed(data_name, data_path)
    X_all /= X_all.sum(axis=1)[:, None]
    task_type = 'classification' if len(
        np.unique(y_all)) == 2 else 'regression'
    print(f'data_name: {data_name}, task_type: {task_type}')

    # Results path
    folder_name = data_name + '_' + run_name_root + '-' + cv_type
    results_path = exp_path + 'results/' + folder_name

    for seed_num in seed_nums:
        # Load CV idx
        idx_dict_all = np.load(join(results_path,
                                    f'cv_idx_all_seed{seed_num}.npy'),
                               allow_pickle='TRUE')

        # Load for fold 0
        fold_idx = 0
        # ytrue
        y_te = y_all[idx_dict_all[fold_idx]['te']]
        # yscore
        yscore_all = pd.read_csv(join(
            results_path,
            f'yscore_fold{fold_idx}_seed{seed_num}.csv'))
        yscore_all['y'] = y_te
        yscore_all['fold'] = fold_idx
        # yhat
        yhat_all = pd.read_csv(join(
            results_path,
            f'yhat_fold{fold_idx}_seed{seed_num}.csv'))
        yhat_all['y'] = y_te
        yhat_all['fold'] = fold_idx
        # best kernels for fold_idx=0
        best_kernels_all = pd.read_csv(join(
            results_path,
            f'best_kernels_fold{fold_idx}_seed{seed_num}.csv'))
        best_kernels_all['fold'] = fold_idx

        # Iterate over remaining folds
        for fold_idx in range(1, n_fold):
            print(f'#### fold_idx: {fold_idx} ####')
            try:
                # ytrue
                y_te = y_all[idx_dict_all[fold_idx]['te']]
                # yscore
                yscore_new = pd.read_csv(join(
                    results_path,
                    f'yscore_fold{fold_idx}_seed{seed_num}.csv'))
                yscore_new['y'] = y_te
                yscore_new['fold'] = fold_idx
                yscore_all = pd.concat([yscore_all, yscore_new])
                # yhat
                yhat_new = pd.read_csv(join(
                    results_path,
                    f'yhat_fold{fold_idx}_seed{seed_num}.csv'))
                yhat_new['y'] = y_te
                yhat_new['fold'] = fold_idx
                yhat_all = pd.concat([yhat_all, yhat_new])
                # best kernels
                best_kernels_new = pd.read_csv(join(
                    results_path,
                    f'best_kernels_fold{fold_idx}_seed{seed_num}.csv'))
                best_kernels_new['fold'] = fold_idx
                best_kernels_all = pd.concat([best_kernels_all,
                                              best_kernels_new])
            except Exception:
                print(f"fold_idx {fold_idx} not found.")
                pass

        yscores[seed_num] = yscore_all
        yhats[seed_num] = yhat_all
        best_kernels[seed_num] = best_kernels_all

    return yscores, yhats, best_kernels, task_type


def plot_roc_curves(ax, yscores, ylab, xlab):

    # method names
    list(yscores[seed_nums[0]].keys()[:-2])
    # create plot
    for jj, mm in enumerate(methods):
        # adapted from:
        # https://towardsdatascience.com/pooled-roc-with-xgboost-and-plotly-553a8169680c
        fpr_mean = np.linspace(0, 1, 100)  # for x-axis
        interp_tprs = []
        for kk, seed_num in enumerate(seed_nums):
            fpr, tpr, thresholds = roc_curve(
                yscores[seed_num]['y'],
                return_rank(yscores[seed_num][mm]),
                drop_intermediate=False)
            interp_tpr = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        alpha = 0.05
        tpr_lwr = np.quantile(interp_tprs, alpha, axis=0)
        tpr_upr = np.quantile(interp_tprs, 1-alpha, axis=0)
        ax.plot(fpr_mean, tpr_mean,
                label=mm, color=ccodes[cnames[jj]],
                linewidth=0.5)
        ax.fill(np.append(fpr_mean, fpr_mean[::-1]),
                np.append(tpr_lwr, tpr_upr[::-1]),
                alpha=0.2, color=ccodes[cnames[jj]])
    if xlab:
        ax.set_xlabel('False positive rate')
    if ylab:
        ax.set_ylabel('True positive rate')
    # adjustments to data_name
    if data_name == 'genomemed_cancer':
        ax.set_title('genomemed', pad=3.0)
    elif data_name == 'americangut_uk':
        ax.set_title('americangut', pad=3.0)
    elif data_name == 'centralparksoil':
        ax.set_title('centralpark', pad=3.0)
    else:
        ax.set_title(data_name, pad=3.0)
    # further adjustments
    ax.tick_params(left=False, bottom=False)
    ax.spines['top'].set(color='gray', linewidth=0.5, alpha=0.2)
    ax.spines['right'].set(color='gray', linewidth=0.5, alpha=0.2)
    ax.spines['left'].set(color='gray', linewidth=0.5, alpha=0.2)
    ax.spines['bottom'].set(color='gray', linewidth=0.5, alpha=0.2)
    ax.tick_params(axis='y', which='major', pad=-2)
    # axis ticks sizes
    ax.tick_params(axis='y', labelsize=fontsize*0.65)
    ax.yaxis.offsetText.set_fontsize(fontsize*0.65)
    ax.tick_params(axis='x', labelsize=fontsize*0.65)
    ax.yaxis.offsetText.set_fontsize(fontsize*0.65)


def plot_summary(ax, yhats, ylab):

    # method names
    methods = list(yhats[seed_nums[0]].keys()[:-2])
    # score function
    score_fun = mean_squared_error if task_type == 'regression' \
        else accuracy_score
    # per fold score
    scores = {}
    for seed_num in seed_nums:
        for mm in methods:
            if mm not in scores.keys():
                scores[mm] = []
            score_tmp = score_fun(yhats[seed_num]['y'],
                                  yhats[seed_num][mm])
            scores[mm].append(score_tmp)
    scores = pd.DataFrame(scores)

    # prepare boxplot
    names, vals, xs = [], [], []
    for i, mm in enumerate(methods):
        names.append(mm)
        vals.append(scores[mm].tolist())
        xs.append(np.random.normal(i+1, 0.12, scores.shape[0]))
    # create boxplot
    ngroup = len(vals)
    for x, val, jj in zip(xs, vals, list(range(ngroup))):
        ax.scatter(x, val, s=1.5, c=ccodes[cnames[jj]], alpha=0.5)
    boxprops = dict(linewidth=0.2, color='black')
    whiskerprops = dict(linewidth=0.2, color='black')
    capprops = dict(linewidth=0.2, color='black')
    medianprops = dict(linewidth=0.2, color='black')
    box_plot = ax.boxplot(vals,
                          boxprops=boxprops,
                          capprops=capprops,
                          whiskerprops=whiskerprops,
                          medianprops=medianprops,
                          showfliers=False, widths=0.8,
                          patch_artist=False)
    for median in box_plot['medians']:
        median.set_color('black')

    # update yaxis
    if task_type == 'regression':
        ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=fontsize*0.65)
    ax.yaxis.offsetText.set_fontsize(fontsize*0.65)

    # add labels
    if ylab:
        if task_type == 'regression':
            ax.set_ylabel('MSE')
        else:
            ax.set_ylabel('accucary')
    # adjustments to data_name
    if data_name == 'genomemed_cancer':
        ax.set_title('genomemed', pad=3.0)
    elif data_name == 'americangut_uk':
        ax.set_title('americangut', pad=3.0)
    elif data_name == 'centralparksoil':
        ax.set_title('centralpark', pad=3.0)
    else:
        ax.set_title(data_name, pad=3.0)
    # further adjustments
    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    ax.tick_params(left=False, bottom=False, labelbottom=False)
    ax.spines['top'].set(color='gray', linewidth=0.5, alpha=0.2)
    ax.spines['right'].set(color='gray', linewidth=0.5, alpha=0.2)
    ax.spines['left'].set(color='gray', linewidth=0.5, alpha=0.2)
    ax.spines['bottom'].set(color='gray', linewidth=0.5, alpha=0.2)
    ax.tick_params(axis='y', which='major', pad=-2.0)


def create_score_tbls(methods, yhats, anres_path):

    score_fun = mean_squared_error if task_type == 'regression' \
        else accuracy_score
    score_mean = []
    score_lwr = []
    score_upr = []
    alpha = 0.05
    for jj, mm in enumerate(methods):
        score_mm = []
        for kk, seed_num in enumerate(seed_nums):
            score_tmp = score_fun(
                yhats[seed_num]['y'], yhats[seed_num][mm])
            score_mm.append(score_tmp)
        score_mean.append(np.median(score_mm, axis=0))
        score_lwr.append(np.quantile(score_mm, alpha, axis=0))
        score_upr.append(np.quantile(score_mm, 1-alpha, axis=0))

    score_tbl = pd.DataFrame({'Mean': score_mean,
                              'LWR': score_lwr,
                              'UPR': score_upr})
    score_tbl.index = methods
    score_tbl.sort_values(by='Mean', ascending=False, inplace=True)
    score_tbl.to_csv(
        join(anres_path, f'{data_name}_{run_name_root}_score_allseeds_cr.csv'),
        float_format='%.3f')


def create_best_kernel_tbls(best_kernels, anres_path, n_folds=10):

    selection_prob = {'linear': 0,
                      'rbf': 0,
                      'aitchison': 0,
                      'aitchison-rbf': 0,
                      'hilbertian': 0,
                      'generalized-js': 0,
                      'heat-diffusion': 0}
    mm = len(selection_prob)
    for kk, seed_num in enumerate(seed_nums):
        best_mods = [ss.split('_', maxsplit=1)[0]
                     for ss in best_kernels[
            seed_num]['estimator_key'][::mm]]
        for kernel in selection_prob.keys():
            selection_prob[kernel] += np.sum([kernel == bb
                                              for bb in best_mods])
    for kernel in selection_prob.keys():
        selection_prob[kernel] /= len(seed_nums)*n_folds
    selection_prob = pd.DataFrame(selection_prob, index=[0])
    selection_prob.to_csv(
        join(anres_path,
             f'{data_name}_{run_name_root}_selection_prob_kernels.csv'),
        float_format='%.3f')


##
# go through datasets
##

# Figure for summary plot
fig, ax = plt.subplots(grid_dim_x, grid_dim_y)
fig_roc, ax_roc = plt.subplots(grid_dim_x_roc, grid_dim_y_roc, squeeze=False)
kkk = 0
for kk, data_name in enumerate(datasets):

    yscores, yhats, best_kernels, task_type = gather_results(
        data_name, data_path, exp_path, run_name_root,
        seed_nums, cv_type='kfold', n_fold=10)

    # Create directory (if it doesnt exist)
    anres_path = exp_path + f"analyzed_results/{data_name}_{run_name_root}/"
    os.makedirs(anres_path, exist_ok=True)

    # Extract method names
    methods = list(yscores[seed_nums[0]].keys()[:-2])

    # Create roc plots
    if task_type == 'classification':
        yy = kkk % grid_dim_y_roc
        xx = int(kkk / grid_dim_y_roc)
        plot_roc_curves(ax_roc[xx, yy], yscores,
                        yy == 0, xx == grid_dim_x_roc-1)
        kkk += 1

    # Create summary plot
    yy = kk % grid_dim_y
    xx = int(kk / grid_dim_y)
    plot_summary(ax[xx, yy], yhats, yy == 0)
    # Create score tables
    create_score_tbls(methods, yhats, anres_path)
    # Create selection probabilities of kernels
    create_best_kernel_tbls(best_kernels, anres_path)

# Construct legend
# modify 'KB_Aitchison' to 'KB-Aitchison' in the legend
legend_lbls = methods.copy()
legend_lbls[-1] = 'KB-Aitchison'
legend_handle = []
for jj, mm in enumerate(methods):
    legend_marker = mlines.Line2D([], [], color=ccodes[cnames[jj]],
                                  marker='o', linestyle='None', alpha=0.5,
                                  markersize=3, label=mm)
    legend_handle.append(legend_marker)
fig.legend(handles=legend_handle,
           labels=legend_lbls,
           loc='lower center',
           labelspacing=0.2,
           columnspacing=0.5,
           bbox_to_anchor=(0.5, -0.15),
           handletextpad=0.02,  # space between marker and text
           fancybox=False,
           shadow=False, ncol=4,
           frameon=False)
fig.subplots_adjust(bottom=0.20, wspace=0.23)

fig_roc.legend(handles=legend_handle,
               labels=legend_lbls,
               loc='lower center',
               labelspacing=0.1,
               columnspacing=0.5,
               bbox_to_anchor=(0.51, -0.15),
               handletextpad=0.02,  # space between marker and text
               fancybox=False,
               shadow=False, ncol=7,
               frameon=False)
fig_roc.subplots_adjust(bottom=0.15, wspace=0.18)

# Save figures (curated)
fig_name = 'curated'
fig.set_size_inches(width, width/2)
fig.tight_layout(pad=0.8)
fig.savefig(join(
    exp_path + '/analyzed_results/',
    f'{fig_name}_summary_plot.pdf'),
    bbox_inches='tight')

fig_roc.set_size_inches(2*width, width/2)
fig_roc.tight_layout(pad=0.8)
fig_roc.savefig(join(
    exp_path + '/analyzed_results/',
    f'{fig_name}_roc_plot.pdf'),
    bbox_inches='tight')
