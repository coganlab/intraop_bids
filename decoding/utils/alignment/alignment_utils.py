"""Alignment utility functions.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
import pickle
from functools import reduce


def extract_group_conditions(Xs, ys):
    ys = [label2str(labs) for labs in ys]

    cnd_avg_data = [0] * len(Xs)
    for i, (feats, labs) in enumerate(zip(Xs, ys)):
        cnd_avg_data[i] = cnd_avg(feats, labs)

    shared_lab = reduce(np.intersect1d, ys)
    cnd_avg_data = [cnd_avg_data[i][np.isin(np.unique(lab), shared_lab,
                                            assume_unique=True)]
                    for i, lab in enumerate(ys)]
    return cnd_avg_data


def cnd_avg(data, labels):
    """Average data trials along first axis by condition label.

    Args:
        data: shape (n_trials, ...)
        labels: shape (n_trials,)

    Returns:
        shape (n_conditions, ...)
    """
    data_shape = data.shape
    class_shape = (len(np.unique(labels)),) + data_shape[1:]
    data_by_class = np.zeros(class_shape)
    for i, seq in enumerate(np.unique(labels)):
        data_by_class[i] = np.mean(data[labels == seq], axis=0)
    return data_by_class


def label2str(labels):
    if len(labels.shape) > 1:
        labels = label_seq2str(labels)
    else:
        labels = labels.astype(str)
    return labels


def label_seq2str(labels):
    """Convert 2D label sequences to 1D label strings.

    E.g. [1, 2, 3] -> '123'. Used for phoneme sequence labels.
    """
    labels_str = []
    for i in range(labels.shape[0]):
        labels_str.append(''.join(str(x) for x in labels[i, :]))
    return np.array(labels_str)


def save_pkl(data, filename):
    with open(filename, 'wb+') as f:
        pickle.dump(data, f, protocol=-1)


def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def decoding_data_from_dict(data_dict, pt, p_ind, lab_type='phon',
                            algn_type='phon_seq'):
    D_tar, lab_tar, lab_tar_full = get_features_labels(data_dict[pt], p_ind,
                                                       lab_type, algn_type)

    pre_data = []
    for p_pt in data_dict[pt]['pre_pts']:
        D_curr, lab_curr, lab_curr_full = get_features_labels(data_dict[p_pt],
                                                              p_ind, lab_type,
                                                              algn_type)
        pre_data.append((D_curr, lab_curr, lab_curr_full))

    return (D_tar, lab_tar, lab_tar_full), pre_data


def get_features_labels(data, p_ind, lab_type, algn_type):
    lab_full = data['y_full_' + algn_type[:-4]]
    if p_ind == -1:
        D = data['X_collapsed']
        lab = data['y_' + lab_type + '_collapsed']
        lab_full = np.tile(lab_full, (3, 1))
    else:
        D = data['X' + str(p_ind)]
        lab = data['y' + str(p_ind)]
    if lab_type == 'artic':
        lab = phon_to_artic_seq(lab)
    return D, lab, lab_full


def phon_to_artic_seq(phon_seq):
    phon_to_artic_conv = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4,
                          9: 4}
    flat_seq = phon_seq.flatten()
    artic_conv = np.array([phon_to_artic(phon_idx, phon_to_artic_conv)
                           for phon_idx in flat_seq])
    return np.reshape(artic_conv, phon_seq.shape)


def phon_to_artic(phon_idx, phon_to_artic_conv):
    return phon_to_artic_conv[phon_idx]
