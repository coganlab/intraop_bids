""" Metrics for comparing quality of different cross-patient dataset alignment
methods.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
import scipy.stats as stats
from .alignment_utils import extract_group_conditions


# def pt_corr_multi(target, to_corr_list, p_vals=False):
#     pt_corrs = [0]*len(to_corr_list)
#     pt_p_vals = [0]*len(to_corr_list)
#     for i, to_corr in enumerate(to_corr_list):
#         if p_vals:
#             pt_corrs[i], pt_p_vals[i] = pt_corr(target, to_corr, p_vals=True)
#         else:
#             pt_corrs[i] = pt_corr(target, to_corr)
#     if p_vals:
#         return pt_corrs, pt_p_vals
#     return pt_corrs


# def pt_corr(target, to_corr, p_vals=False):
#     cnd_r = np.zeros(target.shape[0])
#     cnd_p_vals = np.zeros(target.shape[0])
#     for cnd in range(target.shape[0]):
#         curr_cnd_tar = target[cnd].flatten()  # (time * PCs)
#         curr_cnd_corr = to_corr[cnd].flatten()
#         cnd_r[cnd] = pearsonr(curr_cnd_tar, curr_cnd_corr)[0]
#         cnd_p_vals[cnd] = pearsonr(curr_cnd_tar, curr_cnd_corr)[1]
#     if p_vals:
#         return cnd_r, cnd_p_vals
#     return cnd_r  # return average r value across conditions

def pt_corr(target, to_corr, target_labels, to_corr_labels, n_comp=25):
    min_dim = np.min([target.shape[-1], to_corr.shape[-1], n_comp])
    cnd_avg_data = extract_group_conditions([target, to_corr], [target_labels, to_corr_labels])
    n_cnds = cnd_avg_data[0].shape[0]
    cnd_r = np.zeros(n_cnds)
    for i in range(cnd_avg_data[0].shape[0]):
        target_avg = cnd_avg_data[0][i,:,:min_dim]
        to_corr_avg = cnd_avg_data[1][i,:,:min_dim]

        r_vals = np.zeros(min_dim)
        for j in range(min_dim):
            r_vals[j] = stats.pearsonr(target_avg[:,j], to_corr_avg[:,j])[0]
        cnd_r[i] = np.mean(r_vals)
    return cnd_r

def pt_corr_multi(target, to_corr_list, target_labels, to_corr_labels_list, n_comp=25):
    cnd_r = []
    for i, to_corr in enumerate(to_corr_list):
        cnd_r.append(pt_corr(target, to_corr, target_labels, to_corr_labels_list[i], n_comp=n_comp))
    return cnd_r
