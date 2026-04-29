"""Metrics for comparing cross-patient dataset alignment quality.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
import scipy.stats as stats
from .alignment_utils import extract_group_conditions


def pt_corr(target, to_corr, target_labels, to_corr_labels, n_comp=25):
    min_dim = np.min([target.shape[-1], to_corr.shape[-1], n_comp])
    cnd_avg_data = extract_group_conditions(
        [target, to_corr], [target_labels, to_corr_labels])
    n_cnds = cnd_avg_data[0].shape[0]
    cnd_r = np.zeros(n_cnds)
    for i in range(n_cnds):
        target_avg = cnd_avg_data[0][i, :, :min_dim]
        to_corr_avg = cnd_avg_data[1][i, :, :min_dim]

        r_vals = np.zeros(min_dim)
        for j in range(min_dim):
            r_vals[j] = stats.pearsonr(target_avg[:, j],
                                       to_corr_avg[:, j])[0]
        cnd_r[i] = np.mean(r_vals)
    return cnd_r


def pt_corr_multi(target, to_corr_list, target_labels, to_corr_labels_list,
                  n_comp=25):
    cnd_r = []
    for i, to_corr in enumerate(to_corr_list):
        cnd_r.append(pt_corr(target, to_corr, target_labels,
                             to_corr_labels_list[i], n_comp=n_comp))
    return cnd_r
