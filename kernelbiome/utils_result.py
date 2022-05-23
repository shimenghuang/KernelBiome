import pandas as pd
from collections import Counter
from .kernels_jax import *


# ---- helpers for using the return value of top_models_in_each_group ----


def get_internel_kernel_name(kernel_name):
    """
    Map kernel names to internel kernel names used in `k_<internel_name>` and `kmat_<internel_name>`.
    """
    kernel_names_dict = {
        'linear': 'linear',
        'rbf': 'rbf',
        'generalized-js': 'hilbert1',
        'hilbertian': 'hilbert2',
        'aitchison': 'aitchison',
        'aitchison-rbf': 'aitchison_rbf',
        'heat-diffusion': 'hd'
    }
    return kernel_names_dict[kernel_name]


def kernel_args_str_to_dict(kernel_args_key, weighted=False):
    """
    Convert a concatenated string of kernel argument e.g. 'aitchison-rbf_c_0.0001_g_0.1' to a dict.
    """
    if weighted:
        split_res = kernel_args_key.split('_weighted_', maxsplit=1)
    else:
        split_res = kernel_args_key.split('_', maxsplit=1)
    if len(split_res) == 1:
        return {}
    else:
        kernel_args_str = split_res[1]
        args_list = kernel_args_str.split('_')
        # after removing the name of the kernel the number of elements should be even
        assert(len(args_list) % 2 == 0)
        kernel_args_dict = {}
        for key, val in zip(args_list[0::2], args_list[1::2]):
            kernel_args_dict[key] = float(val)
        return kernel_args_dict


def kernel_args_str_to_k_fun(kernel_args_key, weighted=False, w_mat=None):
    """
    Convert a concatenated string of kernel argument to the corresponding kernel function `k_<internel_kernel_name>`.

    e.g. 'aitchison-rbf_c_0.0001_g_0.1' to a warpped function that takes two vectors x and y only.
    """
    internel_kernel_name = get_internel_kernel_name(
        kernel_args_key.split('_', maxsplit=1)[0])
    kernel_args_dict = kernel_args_str_to_dict(
        kernel_args_key, weighted=weighted)
    if weighted:
        return lambda x, y: eval('k_'+internel_kernel_name+'_weighted')(x, y, w=w_mat, **kernel_args_dict)
    else:
        return lambda x, y: eval('k_'+internel_kernel_name)(x, y, **kernel_args_dict)


# ---- pipeline utilities ----

def make_result_table(mod_with_params, train_scores_all, test_scores_all, selected_params_all):
    kmat_and_params = mod_with_params.keys()
    kmat_and_params = [ss.split('_', maxsplit=1) for ss in kmat_and_params]
    kmat_and_params = pd.DataFrame(kmat_and_params, columns=[
                                   'kernel', 'kernel_params'])
    # index = pd.MultiIndex.from_frame(kmat_and_params)
    res = pd.DataFrame({'avg_train_score': train_scores_all.mean(axis=1),
                        'avg_test_score': test_scores_all.mean(axis=1),
                        'most_freq_best_param': [Counter(row).most_common(1)[0][0] for row in selected_params_all],
                        # 'best_test_score': test_scores_all.max(axis=1),
                        # 'best_test_param': [row[ii] for row, ii in zip(selected_params_all, np.argmax(test_scores_all, axis=1))]
                        })
    res = pd.concat([kmat_and_params, res], axis=1)
    res.sort_values(['kernel', 'avg_test_score'],
                    ascending=[True, False], inplace=True)
    res.set_index(['kernel', 'kernel_params'], inplace=True)
    return res


def top_models_in_each_group(mod_with_params, train_scores_all, test_scores_all, selected_params_all, top_n=1, kernel_mod_only=False):
    estimator_keys = list(mod_with_params.keys())
    kmat_and_params = [ss.split('_', maxsplit=1) for ss in estimator_keys]
    kmat_and_params = pd.DataFrame(kmat_and_params, columns=[
                                   'kernel', 'kernel_params'])
    res = pd.DataFrame({'estimator_key': estimator_keys,
                        'kmat_fun': mod_with_params.values(),
                        'avg_train_score': train_scores_all.mean(axis=1),
                        'avg_test_score': test_scores_all.mean(axis=1),
                        'most_freq_best_param': [Counter(row).most_common(1)[0][0] for row in selected_params_all]
                        })
    res = pd.concat([kmat_and_params, res], axis=1)
    # drop RF, baseline, linear and rbf if required
    if kernel_mod_only:
        rf_idx = [ii for ii in range(len(
            estimator_keys)) if 'RF' in estimator_keys[ii] or 'baseline' in estimator_keys[ii]]
        res.drop(rf_idx, inplace=True)
    # get top_n models in each group, models[0].kernel will give the kernel
    res.sort_values(['kernel', 'avg_test_score', 'avg_train_score'], ascending=[
                    True, False, False], inplace=True)
    top_res = res.groupby('kernel')[
        ['estimator_key', 'kmat_fun', 'avg_test_score', 'most_freq_best_param']].head(top_n)
    return top_res.sort_values('avg_test_score', ascending=False)
