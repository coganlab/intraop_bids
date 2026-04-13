"""Shared preprocessing functions for IEEG data.

Provides the common preprocessing pipeline steps used across epoch extraction
and visualization scripts: event extraction, bad channel handling, powerline
noise removal, re-referencing, and trial quality control.
"""

import logging

import mne
import numpy as np

from utils.stats import find_channel_outliers, remove_bad_trials
from utils.referencing import (set_white_matter_reference,
                               set_laplacian_reference)
from utils.dataloaders import load_raw

logger = logging.getLogger(__name__)


def get_events(raw, filter_str):
    """Extract events from raw annotations matching a filter string."""
    logger.info('Getting events from annotations with filter string: '
                f'{filter_str}')
    events, event_id = mne.events_from_annotations(raw)
    sel_events = [e for e in event_id.keys() if filter_str in e]
    logger.info(f'Found events: {sel_events}')
    return sel_events, event_id


def set_bad_channels(raw, bad_channels):
    """Mark channels as bad on the raw object."""
    logger.info(f'Adding bad channels to raw data: {bad_channels}')
    bad_channels_tot = list(set(raw.info['bads'] + bad_channels))
    raw.info['bads'] = bad_channels_tot
    logger.info(f'New bad channels: {raw.info["bads"]}')


def remove_powerline_noise(raw):
    """Apply notch filter at powerline frequency and harmonics."""
    logger.info('Removing powerline noise via notch filter at powerline '
                'frequency and harmonics')
    powerline_freq = raw.info.get("line_freq")
    if powerline_freq is None:
        logger.warning('Line frequency not found in raw.info, using 60 Hz')
        powerline_freq = 60
    else:
        logger.info(f'Using detected powerline frequency {powerline_freq} Hz')
    freqs = [powerline_freq * m for m in range(1, 4)]
    raw.notch_filter(freqs=freqs, notch_widths=2, n_jobs=-1)


def set_reference(raw, ref='CAR'):
    """Apply a re-referencing scheme to raw data.

    Supported schemes: CAR (common average), WM (white matter), LAP (Laplacian).
    """
    logger.info(f'Setting re-reference scheme: {ref}')
    if ref.lower() in ['car', 'average', 'common_average', 'common average']:
        ch_type = raw.get_channel_types(only_data_chs=True)[0]
        raw = raw.set_eeg_reference(ref_channels="average", ch_type=ch_type)
    elif ref.lower() in ['wm', 'white_matter', 'white matter']:
        raw = set_white_matter_reference(raw)
    elif ref.lower() in ['lap', 'laplacian']:
        raw = set_laplacian_reference(raw)
    else:
        logger.error(f'Unknown reference: {ref}')
        raise ValueError(f"Unknown reference: {ref}")
    return raw


def get_good_trials(data, threshold=10, method=1, chan_thresh=0.8):
    """Identify good trials shared across channels.

    Returns a boolean mask over trials that are good on at least
    ``chan_thresh`` fraction of channels.
    """
    _, good_trials = remove_bad_trials(data, threshold=threshold,
                                       method=method)

    nChans = data.shape[1]
    assert good_trials.shape[0] == nChans
    good_trials_common = np.sum(good_trials, axis=0) >= (chan_thresh * nChans)

    return good_trials_common


def preprocess_raw(bids_layout, patient, reference='CAR', load_kwargs=None):
    """Run the full preprocessing chain on a BIDS raw recording.

    Steps: load -> outlier detection -> notch filter -> re-reference.

    Returns the preprocessed Raw object.
    """
    if load_kwargs is None:
        load_kwargs = {}

    raw = load_raw(bids_layout, patient, **load_kwargs)

    bad_channels_detrend = find_channel_outliers(raw)
    logger.info(f"IEEG outlier channels (detrend): {bad_channels_detrend}")
    set_bad_channels(raw, bad_channels_detrend)

    ch_type = raw.get_channel_types(only_data_chs=True)[0]
    logger.info(f'Available channel types: {ch_type}')

    remove_powerline_noise(raw)

    raw = set_reference(raw, reference)

    return raw
