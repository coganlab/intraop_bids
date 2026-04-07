"""Extract epochs from BIDS iEEG recordings with preprocessing and high-gamma features.

Purpose:
- Load iEEG data from a BIDS layout and derive per-trial epochs.
- Ensure trial consistency by removing incomplete/mismatched events via `remove_missing_events()`.
- Apply standard preprocessing (e.g., scaling and artifact handling utilities from `ieeg`).
- Compute high-gamma features using `ieeg.timefreq.gamma` for downstream analyses.

Outputs:
- Returns/produces epoched data suitable for modeling or further statistics.
"""
import argparse
from os import path
import pandas as pd

import mne
import numpy as np
from scipy.stats import permutation_test, false_discovery_control, norm
from scipy.signal import detrend
from mne_bids import BIDSPath, read_raw_bids
from mne_bids.dig import _read_dig_bids
from mne_bids.path import _find_matching_sidecar
from ieeg.navigate import channel_outlier_marker
from ieeg.navigate import outliers_to_nan
from ieeg.calc.scaling import rescale
from ieeg.timefreq import gamma
from ieeg.io import raw_from_layout
from bids.layout import BIDSLayout
from pathlib import Path
from ieeg.navigate import trial_ieeg
from mne_bids import get_bids_path_from_fname
import re
import logging
import sys
from joblib import Parallel, delayed

# Simple logging: everything INFO and above to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def interpolate(epochs, min_trials_per_class=2):
    """
    Fill NaNs per channel and class using class/global mean & std.

    - For each channel and class:
      * If class has >= min_trials_per_class valid samples -> fill with class mean (or mean+noise*std)
      * Else -> fill with global mean (or mean+noise*std)
    - If global has no donors at all (extremely rare), fill 0.0

    Parameters
    ----------
    epochs : mne.Epochs
    min_trials_per_class : int
        Threshold for using class-specific statistics
    """
    data = epochs._data  # shape (n_epochs, n_channels, n_times)
    if not np.any(np.isnan(data)):
        print("No NaN values found, skipping interpolation")
        return

    n_epochs, n_channels, n_times = data.shape
    cond_labels = epochs.events[:, 2]
    unique_classes = np.unique(cond_labels)

    total_nans = int(np.isnan(data).sum())
    print(f"Interpolating {total_nans} NaN values ({total_nans/(n_epochs*n_channels*n_times)*100:.2f}% of data)")

    for ch in range(n_channels):
        channel = data[:, ch, :]  # view (epochs, times)

        # get nonnan trials
        nan_trials = np.any(np.isnan(channel), axis=1)
        global_valid = channel[~nan_trials]
        g_mean = np.mean(global_valid, axis=0)
        g_std  = np.std(global_valid, axis=0)
        
        for c in unique_classes:
            rows = np.where(cond_labels == c)[0]
            sub = channel[rows, :]                 # (n_rows, times)
            sub_nan_trials = np.any(np.isnan(sub), axis=-1)

            # if no nan trials continue
            if np.sum(sub_nan_trials)==0:
                continue
            
            # Class donors for this channel (all non-NaN samples in this class)
            class_valid = sub[~sub_nan_trials]
            n_class_valid = class_valid.shape[0]

            if n_class_valid >= min_trials_per_class:
                c_mean = np.mean(class_valid, axis=0)
                c_std  = np.std(class_valid, axis=0)
                mean_to_use, std_to_use = c_mean, c_std
            else:
                mean_to_use, std_to_use = g_mean, g_std

            # Prepare replacement values for all NaNs in this class
            N = class_valid.shape[-1]
            
            for k, nan_trial in enumerate(sub_nan_trials):
                if nan_trial:
                    channel[rows[k], :] = mean_to_use + np.random.randn(N) * 1e-2 * std_to_use

        # Persist filled channel back
        data[:, ch, :] = channel
    epochs._data = data
    print(f"Interpolation complete. Remaining NaNs: {int(np.isnan(data).sum())}")
    
    return

def set_laplacian_reference(
    raw: mne.io.Raw
):
    """
    Apply a 1D Laplacian/bipolar re-reference along each electrode shaft.

    Overview
    - Channels are grouped by shaft using a name pattern like "LA1, LA2, ..." or
      "E1, E2, ...". The regex `([a-zA-Z]+)(\d+)` extracts:
        - shaft name: alphabetic prefix (e.g., "LA", "E")
        - contact index: numeric suffix (e.g., 1, 2, ...)
    - For each shaft with >= 2 contacts:
        - First contact: V1 - V2
        - Last contact: VN - V(N-1)
        - Middle contacts: Vk - 0.5 * (V(k-1) + V(k+1)) (discrete Laplacian)

    Parameters
    - raw (mne.io.BaseRaw): MNE Raw object containing iEEG channels. Channel names
      are expected to follow the pattern described above so that grouping works.

    Returns
    - mne.io.BaseRaw: A COPY of the input with Laplacian-referenced data.

    Assumptions & Caveats
    - Channel naming must be consistent and monotonic along the shaft, e.g., LA1, LA2, ...
    - This function accesses `raw._data` directly for speed; this bypasses some of MNE's
      provenance handling. Use with care and only on loaded data (`raw.load_data()`).
    - Sorting: channel order within a shaft should be numerical (1,2,...). If channel names
      do not sort numerically by their numeric suffix, adjust the sorting logic accordingly.
    - Non-iEEG channels should be removed before calling this function.
    - Shafts with <2 contacts are skipped.

    Note on sorting implementation
    - The intent is to sort by contact number (numeric suffix). If your data structure here is a
      list of strings (channel names), consider sorting by the extracted integer index from the
      regex, e.g., `key=lambda name: int(re.match(r'([A-Za-z]+)(\d+)', name).group(2))`.
    """
    # Make a copy to avoid modifying original data
    raw = raw.copy()
    # Group channels by shafts
    shaft_groups = {}
    pattern = re.compile(r'([a-zA-Z]+)(\d+)')
    for ch_name in raw.ch_names:
        match = pattern.match(ch_name)
        if match:
            shaft_name = match.group(1) # The letter part is the shaft name
            contact_num = int(match.group(2))
            if shaft_name not in shaft_groups:
                shaft_groups[shaft_name] = []
            # Collect channel name under its shaft
            shaft_groups[shaft_name].append((ch_name, contact_num))
    for shaft_name, shaft_channels in shaft_groups.items():
        if len(shaft_channels) < 2:
            print(f"Skipping shaft '{shaft_name}' because it has fewer than 2 channels.")
            continue
        shaft_channels.sort(key=lambda x: x[1])
        shaft_channel_names = [ch[0] for ch in shaft_channels]
        
        # Store original data for this shaft to avoid in-place modification corruption
        shaft_indices = [raw.ch_names.index(ch_name) for ch_name in shaft_channel_names]
        original_data = raw._data[shaft_indices, :].copy()
        
        for i, current_ch_name in enumerate(shaft_channel_names):
            current_ch_idx = raw.ch_names.index(current_ch_name)
            if i == 0:
                # First contact: subtract next contact (bipolar)
                # V1 <- V1 - V2
                raw._data[current_ch_idx, :] = original_data[i, :] - original_data[i + 1, :]
            elif i == len(shaft_channel_names) - 1:
                # Last contact: subtract previous contact (bipolar)
                # VN <- VN - V(N-1)
                raw._data[current_ch_idx, :] = original_data[i, :] - original_data[i - 1, :]
            else:
                # Middle contacts: Laplacian (subtract average of neighbors)
                # Vk <- Vk - 0.5 * (V(k-1) + V(k+1))
                raw._data[current_ch_idx, :] = original_data[i, :] - 0.5 * (original_data[i - 1, :] + original_data[i + 1, :])
    return raw


def set_white_matter_reference(
    raw: mne.io.Raw
):
    """
    Set a white-matter (WM) reference using channels labeled as white matter.

    Procedure
    - Reads the BIDS `channels.tsv` sidecar corresponding to the current raw file.
    - Selects channels with `status_description == 'pure_white_matter'`.
      If none, falls back to `status_description == 'white_matter'`.
    - Applies MNE's referencing with the selected WM channels as reference.

    Parameters
    - raw (mne.io.Raw): Loaded Raw object with valid `raw.filenames[0]` path.

    Returns
    - mne.io.Raw: New Raw with WM reference applied (MNE returns a new instance).

    Notes
    - `ch_type` is inferred from the first data channel type to match the dataset (ieeg/eeg).
    - Make sure non-data channels (e.g., Trigger) are dropped prior to referencing.
    """
    # Locate the channels.tsv sidecar via BIDS path
    ref_path = get_bids_path_from_fname(raw.filenames[0])
    ref_path = ref_path.copy().update(suffix="channels", extension=".tsv").fpath

    # Read sidecar to discover WM channels
    ref_df = pd.read_csv(ref_path, sep='\t')
    # Prefer pure white matter channels; if none, use all white matter
    wm_channels = ref_df[ref_df['status_description'] == 'pure_white_matter']['name'].tolist()
    # Keep only those present in the current recording
    wm_channels = [ch for ch in wm_channels if ch in raw.ch_names]
    if not wm_channels:
        wm_channels = ref_df[ref_df['status_description'] == 'white_matter']['name'].tolist()

    # Infer channel type for referencing (e.g., 'ieeg' or 'eeg')
    ch_type = raw.get_channel_types(only_data_chs=True)[0]

    # Apply WM reference; MNE returns a new Raw instance
    raw = raw.set_eeg_reference(ref_channels=wm_channels, ch_type=ch_type)

    return raw

def remove_missing_events(
    raw
):

    # this is to make sure the number of perception and production trials are the same
    events, event_id = mne.events_from_annotations(raw)
    
    # Create reverse mapping from event_id values to names
    id_to_name = {v: k for k, v in event_id.items()}
    
    # Find Start event indices to identify trial boundaries
    start_event_id = event_id['Start/Listen/LS']
    start_indices = np.where(events[:, 2] == start_event_id)[0]
        
    # Collect events from complete trials only
    valid_events = []
    complete_trials = 0
    incomplete_trials = 0
    
    for start_idx in start_indices:
        # Check if we have enough events for a complete trial
        if start_idx + 3 >= len(events):
            incomplete_trials += 1
            continue
            
        # Get the 4 consecutive events starting from Start
        trial_events = events[start_idx:start_idx+4]
        trial_event_names = [id_to_name[event_id] for event_id in trial_events[:, 2]]
        
        # Check if this trial has the complete sequence
        has_start = trial_event_names[0] == 'Start/Listen/LS'
        has_audio = 'Word/Audio' in trial_event_names[1] if len(trial_event_names) > 1 else False
        has_go = 'Word/Go' in trial_event_names[2] if len(trial_event_names) > 2 else False
        has_response = 'Word/Response' in trial_event_names[3] if len(trial_event_names) > 3 else False
        
        # Only keep trials with complete sequence
        if has_start and has_audio and has_go and has_response:
            valid_events.extend(trial_events)
            complete_trials += 1
        else:
            incomplete_trials += 1
            print(f"Dropping incomplete trial starting at sample {trial_events[0][0]}: {trial_event_names}")
    
    print(f"Total trials found: {len(start_indices)}")
    print(f"Complete trials kept: {complete_trials}")
    
    if len(valid_events) == 0:
        print("Warning: No complete trials found!")
        return raw
    
    # Convert back to events array
    valid_events = np.array(valid_events)
    
    # Create new annotations with only valid events
    valid_event_names = [id_to_name[event_id] for event_id in valid_events[:, 2]]
    valid_onsets = valid_events[:, 0] / raw.info['sfreq']  # Convert samples to seconds
    valid_durations = np.zeros(len(valid_events))  # MNE annotations need durations
    
    # Create new annotations
    new_annotations = mne.Annotations(
        onset=valid_onsets,
        duration=valid_durations,
        description=valid_event_names
    )
    
    # Replace the raw annotations with filtered ones
    raw.set_annotations(new_annotations)
    
    return raw


def find_channel_outliers(data, threshold=3):
    data = detrend(data, axis=1)  # ensure detrending along time
    R2 = np.square(data)
    R2[R2 == 0] = 1e-9

    sig = np.std(R2, axis=1)
    m, s = norm.fit(sig)
    out1 = np.where(sig > (threshold*s + m))[0]
    mask = np.setdiff1d(np.arange(len(sig)), out1)
    m2, s2 = norm.fit(sig[mask])
    out2 = np.where(sig[mask] > (threshold*s2 + m2))[0]
    chan_outliers = np.sort(np.concatenate([out1, mask[out2]])) + 1
    chan_outliers = [str(chan) for chan in chan_outliers]
    return chan_outliers


def remove_bad_trials(data, threshold=10, method=1):
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



def get_significant_channels(data_interest, data_baseline,
                             tw_interest=[-0.25,0.25],
                             tw_baseline=[-0.5, 0], alpha=0.05):
    """ Identify significant channels by one-sided permutation test of
    data of interest compared to baseline data averaged in specified time
    windows of interest.

    Args:
        data_interest (mne.Epochs): Epochs object containing data for the
            condition of interest.
        data_baseline (mne.Epochs): Epochs object containing baseline data.
        tw_interest (list, optional): Time window of interest in seconds.
            Defaults to [-0.25, 0.25].
        tw_baseline (list, optional): Baseline time window in seconds.
            Defaults to [-0.5, 0].
        alpha (float, optional): Significance level for FDR correction.
            Defaults to 0.05.

    Returns:
        list: List of significant channel names as they appear in the input
            epoch objects for the data of interest.
    
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


def preproc(
    bids_layout: BIDSLayout,
    subject: str,
    useSig: bool,
    tmin: float,
    tmax: float,
    fs: int,
    ref: str,
    **kwargs
):

    # Load data
    raw = raw_from_layout(
        bids_layout,
        subject=subject,
        extension='.edf',
        desc=None,
        **kwargs
    )
    raw.load_data()
    raw._data *= 1e6  # convert to uV
    nChans = raw.info['nchan']

    raw_phon = raw_from_layout(
        bids_layout.derivatives[('derivatives/phonemeLevel')],
        subject=subject,
        extension='.edf',
        desc='phonemeLevel',
        **kwargs
    )
    raw_phon.load_data()
    raw_phon._data *= 1e6  # convert to uV

    events, event_id = mne.events_from_annotations(raw)
    baseline_events = [e for e in event_id.keys() if 'stimulus' in e]
    resp_events = [e for e in event_id.keys() if 'response' in e]

    events_phon, event_id_phon = mne.events_from_annotations(raw_phon)
    production_events = [e for e in event_id_phon.keys() if 'response' in e]

    resp_epochs = trial_ieeg(
        raw,
        event=resp_events,
        times=[-3, 3],
        preload=True,
    )
    # channel outlier removal based on detrended concatenated data
    data_chan_time = resp_epochs.get_data().transpose(1,0,2).reshape(resp_epochs.info['nchan'], -1)
    bad_channels_detrend = find_channel_outliers(data_chan_time)
    print(f"IEEG BAD CHANNELS DETREND PHONEME EPOCHS: {bad_channels_detrend}")
    bad_channels_tot = list(set(raw.info['bads'] + bad_channels_detrend))

    # update bads in raw objects with new outliers
    # raw.info['bads'] = bad_channels_tot
    # raw_phon.info['bads'] = bad_channels_tot
    raw.drop_channels(bad_channels_tot)
    raw_phon.drop_channels(bad_channels_tot)

    # remove Trigger channel we have any
    raw.drop_channels(['Trigger']) if 'Trigger' in raw.ch_names else None
    ch_type = raw.get_channel_types(only_data_chs=True)[0]
    print(f"Available channel types: {ch_type}")

    raw_phon.drop_channels(['Trigger']) if 'Trigger' in raw_phon.ch_names else None
    ch_type = raw_phon.get_channel_types(only_data_chs=True)[0]

    # Notch filter power line noise
    powerline_freq = raw.info.get("line_freq")
    if powerline_freq is None:
        powerline_freq = 60
    freqs = [powerline_freq * m for m in range(1, 4)]
    raw.notch_filter(freqs=freqs, notch_widths=2, n_jobs=-1)
    raw_phon.notch_filter(freqs=freqs, notch_widths=2, n_jobs=-1)

    # Re-reference data

    logger.info(f'Setting re-reference scheme: {ref}')

    if ref == "CAR":
        # CAR will automatically ignore channels marked in bads
        raw = raw.set_eeg_reference(ref_channels="average", ch_type=ch_type)
        raw_phon = raw_phon.set_eeg_reference(ref_channels="average", ch_type=ch_type)
    elif ref == "WM":
        raw = set_white_matter_reference(raw)
        raw_phon = set_white_matter_reference(raw_phon)
    elif ref == "LAP":
        raw = set_laplacian_reference(raw)
        raw_phon = set_laplacian_reference(raw_phon)
    else:
        raise ValueError(f"Unknown reference: {ref}")

    # raw = remove_missing_events(raw)

    # baseline epoch, fixed time window (-0.5s to 0s), adding extra 0.5s padding
    baseline_epoch = trial_ieeg(
        raw,
        event=baseline_events,
        times=[-1, 0.5],
        preload=True,
    )

    # perception_epochs = trial_ieeg(
    #     raw,
    #     event=perception_events,
    #     times=[tmin-0.5, tmax+0.5],
    #     preload=True,
    # )

        # recreate response epochs after filtering and CAR
    resp_epochs = trial_ieeg(
        raw,
        event=resp_events,
        times=[-3, 3],
        preload=True,
    )

    production_epochs = trial_ieeg(
        raw_phon,
        event=production_events,
        times=[tmin-0.5, tmax+0.5],
        preload=True,
    )

    # redefine event_id based on the phonemes (same phonemes at different
    # positions should have the same event_id)
    # get all the keys and sort in alphabetical order
    prod_keys = [remove_arpabet_stress(k.split('/')[-1]) for k in production_epochs.event_id.keys()]
    prod_keys = sorted(list(set(prod_keys)))

    # create dictionary mapping phoneme to event code in alphabetical order
    event_id = {phon: i+1 for i, phon in enumerate(prod_keys)}

    # reverse original event_id to allow mapping from event code to phoneme
    ev2phon = swap_kv_dict(production_epochs.event_id)
    ev2phon = {k: remove_arpabet_stress(v.split('/')[-1]) for k,v in ev2phon.items()}

    # account for merging event ids in the production epochs events
    new_events = []
    for ev in production_epochs.events:
        ev_phon = ev2phon[ev[2]]
        ev[2] = event_id[ev_phon]
        new_events.append(ev)
    production_epochs.event_id = event_id
    production_epochs.events = np.array(new_events)

    # event_id = {k.split('/')[-1]: v for k, v in perception_epochs.event_id.items()}
    # perception_epochs.event_id = event_id

    event_id = {k.split('/')[-1]: v for k, v in baseline_epoch.event_id.items()}
    baseline_epoch.event_id = event_id

    # Removing bad trials on the trial level, not phoneme level
    nPhons = len(production_epochs.events) // len(baseline_epoch.events)
    # tr_remove_data_perc = perception_epochs._data[::nPhons]
    # tr_remove_data_prod = production_epochs._data[::nPhons]
    tr_remove_data_prod = resp_epochs._data

    # Remove bad trials based on derivative thresholding (method 2)
    # _, goodtrials_perception = remove_bad_trials(tr_remove_data_perc, threshold=50, method=2)
    _, goodtrials_production = remove_bad_trials(tr_remove_data_prod, threshold=10, method=1)
    # _, goodtrials_production = remove_bad_trials(tr_remove_data_prod, threshold=50, method=2)
    
    # keep trials that are good on 80% of channels
    nChans = tr_remove_data_prod.shape[1]
    # goodtrials_perception_common = np.sum(goodtrials_perception, axis=0) >= (0.8 * nChans)
    goodtrials_production_common = np.sum(goodtrials_production, axis=0) >= (0.8 * nChans)
    bad_trials = np.where(~goodtrials_production_common)[0]
    print(f"Removing {len(bad_trials)} bad trials in production: {bad_trials}")

    # map back to phoneme level
    # goodtrials_perception_common = np.repeat(goodtrials_perception_common, nPhons)
    goodtrials_production_common = np.repeat(goodtrials_production_common, nPhons)

    # Apply good trials mask
    # perception_epochs = perception_epochs[goodtrials_perception_common]
    production_epochs = production_epochs[goodtrials_production_common]

    # Create output directory
    outpath = BIDSPath(
        root=bids_layout.root+f'/derivatives/epoch(phonemeLevel)({ref})',
        subject=subject,
        task='lexical',
        datatype='epoch(raw)',
        check=False
    )
    outpath.mkdir(exist_ok=True)

    baseline_epoch.save(
        outpath.copy().update(
            suffix="raw",
            extension=".fif",
            description="baseline",
            check=False
        ),
        overwrite=True
    )

    # perception_epochs.save(
    #     outpath.copy().update(
    #         suffix="raw",
    #         extension=".fif",
    #         description="perception",
    #         check=False
    #     ),
    #     overwrite=True
    # )

    production_epochs.save(
        outpath.copy().update(
            suffix="raw",
            extension=".fif",
            description="production",
            check=False
        ),
        overwrite=True
    )

    # Extract frequency bands power
    bands = {
        "highgamma": (70, 150),
        # "theta": (4, 8),
        # "alpha": (8, 13),
        # "beta": (13, 30),
        # "gamma": (30, 65),
        # "low": (2, 40),
    }
    for band, freqs in bands.items():

        # # Phase-ready outputs: bandpass filter only (no Hilbert, no baseline)
        # phase_perception = perception_epochs.copy()
        # phase_production = production_epochs.copy()
        # phase_perception.filter(l_freq=freqs[0], h_freq=freqs[1], n_jobs=-1)
        # phase_production.filter(l_freq=freqs[0], h_freq=freqs[1], n_jobs=-1)

        # # Save bandpass-only epochs
        # phase_out = outpath.copy().update(
        #     datatype='epoch(band)(raw)',
        #     suffix=f"{band}",
        #     extension=".fif",
        #     check=False
        # )
        # phase_out.mkdir(exist_ok=True)
        # phase_perception.save(phase_out.update(description="perception"), overwrite=True)
        # phase_production.save(phase_out.update(description="production"), overwrite=True)
        # print(f"Saved unnormalized {band} epochs to {phase_out}")

        # band_perception = perception_epochs.copy()
        band_production = production_epochs.copy()
        band_baseline = baseline_epoch.copy()

        # gamma.extract(band_perception, passband=freqs, copy=False, n_jobs=-1)
        gamma.extract(band_production, passband=freqs, copy=False, n_jobs=-1)
        gamma.extract(band_baseline, passband=freqs, copy=False, n_jobs=-1)

        # use the pre-stimulus 0.5 s as baseline
        band_baseline = band_baseline.crop(tmin=-0.5, tmax=0)
        # band_perception = band_perception.crop(tmin=tmin, tmax=tmax)
        band_production = band_production.crop(tmin=tmin, tmax=tmax)

        # rescale(band_perception, band_baseline,
        #     mode='zscore',
        #     copy=False)
        band_production_z = rescale(band_production, band_baseline,
                                    mode='zscore',
                                    copy=True)
        band_production_ms = rescale(band_production, band_baseline,
                                     mode='mean',
                                     copy=True)
        

        print(f"Resampling {band} epochs to {fs} Hz")
        # band_perception = band_perception.resample(sfreq=fs, n_jobs=-1)
        band_production = band_production.resample(sfreq=fs, n_jobs=-1)
        band_production_z = band_production_z.resample(sfreq=fs, n_jobs=-1)
        band_production_ms = band_production_ms.resample(sfreq=fs, n_jobs=-1)
        band_baseline = band_baseline.resample(sfreq=fs, n_jobs=-1)
        
        # calculate significant channels if specified
        if useSig:
            print("Calculating significant channels...")
            sig_chs = get_significant_channels(
                # only using response aligned for significance testing
                data_interest=band_production[::nPhons],
                data_baseline=band_baseline,
                tw_interest=[-0.25, 0.25],
                tw_baseline=[-0.5, 0],
                alpha=0.05
            )
            print(f"Found {len(sig_chs)} significant channels: {sig_chs}")

            if len(sig_chs) == 0:
                print("No significant channels found, skipping saving band epochs.")
                continue
        else:
            sig_chs = band_production.ch_names

        # make sure list of sig channels is list of integer-valued strings
        sig_chs = [str(int(ch)) for ch in sig_chs]

        # remove any bad channels from sig_chs if not already removed
        sig_chs = [ch for ch in sig_chs if ch not in bad_channels_tot]
        print(f"After removing bad channels, {len(sig_chs)}  channels remain.")

        # pick only significant channels
        band_production.pick(sig_chs)
        band_production_z.pick(sig_chs)
        band_production_ms.pick(sig_chs)
        band_baseline.pick(sig_chs)

        # Save band-specific epochs with run information
        datatype = 'epoch(band)(power)(sig)' if useSig else 'epoch(band)(power)'
        outpath.update(
            datatype=datatype,
            suffix=f"{band}",
            extension=".fif",
            check=False
        )
        outpath.mkdir(exist_ok=True)
        # band_perception.save(outpath.update(description="perception"), overwrite=True)
        band_production.save(outpath.update(description="production"), overwrite=True)
        band_production_z.save(outpath.update(description="productionZscore"), overwrite=True)
        band_production_ms.save(outpath.update(description="productionMeanSub"), overwrite=True)
        band_baseline.save(outpath.update(description="baseline"), overwrite=True)
        print(f"Saved {band} epochs to {outpath}")


def remove_arpabet_stress(phoneme: str) -> str:
    """Remove stress markers from ARPAbet phonemes.

    E.g., 'AH0' -> 'AH', 'IY1' -> 'IY'
    """
    return re.sub(r'\d', '', phoneme)


def swap_kv_dict(d):
    return dict((v, k) for k, v in d.items())


def main(bids_root: str, subject: str, useSig: bool, tmin: float = -1.0,
         tmax: float = 1.5, fs: int = 200, **kwargs):

    bids_layout = BIDSLayout(
        root=bids_root,
        derivatives=True,
    )
    preproc(
        bids_layout=bids_layout,
        subject=subject,
        useSig=useSig,
        tmin=tmin,
        tmax=tmax,
        fs=fs,
        **kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", default="/hpc/home/zms14/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS", type=str)
    parser.add_argument("--ref", default="CAR", type=str, choices=["CAR"],
                        help='which reference to use, WM is for white matter, CAR is for common average reference, LAP is for Laplacian')
    parser.add_argument("--subject", type=str, default='S41')
    parser.add_argument("--useSig", action='store_true',
                        help="Whether to select significant channels only")
    parser.add_argument("--fs", type=int, default=200, help="Sampling rate for hg, in Hz")
    parser.add_argument("--tmin", type=float, default=-1, help="Start time of epoch relative to event in seconds")
    parser.add_argument("--tmax", type=float, default=1.5, help="End time of epoch relative to event in seconds")

    _args = parser.parse_args()
    main(**vars(_args))

