"""Statistical analysis utilities for neural data processing.

This module provides functions for statistical analysis and quality control
of neural electrophysiology data, including outlier detection, bad trial
identification, and statistical significance testing.

Functions:
    find_channel_outliers: Detect channels with abnormal signal characteristics
    remove_bad_trials: Identify and flag bad trials based on thresholding
    get_significant_channels: Perform statistical testing to find significant
    channels
"""

from typing import List, Tuple

import numpy as np
from scipy.signal import detrend
from scipy.stats import norm, false_discovery_control
import mne
from scipy.stats import permutation_test
from joblib import Parallel, delayed


def find_channel_outliers(raw: mne.io.Raw, threshold: float = 3) -> List[str]:
    """Identify outlier channels based on signal variance statistics.
    
    This function detects channels with unusually high variance by fitting a
    normal distribution to channel-wise standard deviations and identifying
    outliers using a two-stage thresholding approach.
    
    Args:
        raw: MNE raw object with data of shape (channels, time)
        threshold: Number of standard deviations above the mean for outlier detection.
            Defaults to 3.
    
    Returns:
        List of channel numbers (as strings) identified as outliers.
        
    Note:
        Uses a two-stage outlier detection:
        1. Initial outlier detection using all channels
        2. Secondary detection on remaining channels to catch subtle outliers
    """
    # extract data from raw object
    data = raw.get_data()
    ch_names = raw.ch_names

    data = detrend(data, axis=1)  # ensure detrending along time
    R2 = np.square(data)
    R2[R2 == 0] = 1e-9

    sig = np.std(R2, axis=1)
    m, s = norm.fit(sig)
    out1 = np.where(sig > (threshold*s + m))[0]
    mask = np.setdiff1d(np.arange(len(sig)), out1)
    m2, s2 = norm.fit(sig[mask])
    out2 = np.where(sig[mask] > (threshold*s2 + m2))[0]

    # channel outliers as if they were 0-indexed
    chan_outliers = np.sort(np.concatenate([out1, mask[out2]]))

    # invariant to channel naming being 0- or 1-indexed. Iterates over both
    # channel names and index, where index starts at 0 and will always match
    # format of chan_outliers above. Then grabs channel name from index in raw
    # object. If channel names are 1-indexed, these will be different.
    # Otherwise, they are the same
    chan_outliers = [chan_name for i, chan_name in enumerate(ch_names) if i in chan_outliers]
    return chan_outliers


def remove_bad_trials(
    data: np.ndarray, 
    threshold: float = 10, 
    method: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Removes bad trials based on a threshold detection.

    This function evaluates each trial for every channel and removes those
    considered "bad" based on the specified thresholding method.

    Args:
        data (np.ndarray): 
            Array of shape (electrodes, trials, samples) representing the recorded data.
        threshold (float, optional): 
            Threshold for identifying noisy trials. Defaults to 10.
        method (int, optional): 
            Method for thresholding:
            
            - 1: Uses an absolute amplitude threshold based on mean and standard deviation.
            - 2: Uses a differential threshold on the first temporal derivative.
            
            Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - **NumTrials** (`np.ndarray`): Array of shape (electrodes,) containing 
              the number of good trials for each channel.
            - **goodtrials** (`np.ndarray`): Boolean array of shape (electrodes, trials)
              indicating which trials are considered good.

    Raises:
        ValueError: If `method` is not 1 or 2.
    """
    n_trials, n_channels, _ = data.shape
    NumTrials = np.zeros(n_channels, dtype=int)
    goodtrials = np.zeros((n_channels, n_trials), dtype=bool)

    for iCh in range(n_channels):
        tmp = data[:, iCh, :]  # shape: (trials, samples)

        if method == 1:
            th = threshold * np.std(np.abs(tmp)) + np.mean(np.abs(tmp))
            e = np.max(np.abs(tmp), axis=1)
            mask = (e < th) & (e != 0)
        elif method == 2:
            difftmp = np.diff(tmp, axis=1)
            e = np.max(difftmp, axis=1)
            mask = e < threshold
        else:
            raise ValueError("method must be 1 or 2")

        NumTrials[iCh] = np.sum(mask)
        goodtrials[iCh, :] = mask

    return NumTrials, goodtrials



def get_significant_channels(
    data_interest: mne.Epochs, 
    data_baseline: mne.Epochs,
    tw_interest: List[float] = [-0.25, 0.25],
    tw_baseline: List[float] = [-0.5, 0], 
    alpha: float = 0.05
) -> List[str]:
    """Identify significant channels by one-sided permutation test.
    
    Performs statistical comparison between data of interest and baseline data
    using permutation testing with false discovery rate (FDR) correction for
    multiple comparisons across channels.
    
    Args:
        data_interest: Epochs object containing data for the condition of interest.
        data_baseline: Epochs object containing baseline data.
        tw_interest: Time window of interest in seconds for averaging.
            Defaults to [-0.25, 0.25].
        tw_baseline: Baseline time window in seconds for averaging.
            Defaults to [-0.5, 0].
        alpha: Significance level for FDR correction. Defaults to 0.05.

    Returns:
        List of significant channel names as they appear in the input
        epoch objects for the data of interest.
        
    Note:
        Uses parallel processing for permutation tests across channels.
        Applies FDR correction to control for multiple comparisons.
    """
    def mean_stat(X, Y, axis=-1):
        return np.mean(X, axis=axis) - np.mean(Y, axis=axis)

    # average data within specified time windows
    tw_interest = data_interest.time_as_index(tw_interest)
    tw_baseline = data_baseline.time_as_index(tw_baseline)

    data_interest_avg = data_interest._data[:, :, tw_interest[0]:tw_interest[1]].mean(axis=2)
    data_baseline_avg = data_baseline._data[:, :, tw_baseline[0]:tw_baseline[1]].mean(axis=2)

    def compute_pvalue(ch):
        res = permutation_test((data_interest_avg[:, ch],
                                data_baseline_avg[:, ch]),
                               mean_stat,
                               vectorized=True,
                               alternative='greater',
                               random_state=42)
        return res.pvalue
    p_values = Parallel(n_jobs=-1)(delayed(compute_pvalue)(ch) for ch in range(data_interest.info['nchan']))
    p_values = np.array(p_values)

    # FDR correction - channel names are as they appear in data_interest obj
    pvals_corr = false_discovery_control(p_values)
    significant_channels = [data_interest.ch_names[i] for i, p in
                            enumerate(pvals_corr) if p < alpha]

    return significant_channels