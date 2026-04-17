"""
Update events.tsv files in a BIDS dataset from manually corrected .txt files.

Updates both word-level events (in the main BIDS directory) and phoneme-level
events (in ``derivatives/phonemeLevel/``) in a single call.

Steps:
1) Load subject raw from BIDS to obtain the sampling frequency.
2) Update word-level events.tsv from ``--stim_txt`` / ``--resp_txt``.
3) Update phoneme-level events.tsv from ``--stim_phon_txt`` / ``--resp_phon_txt``.
4) Write ``_events.json`` sidecars for both levels.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from bids import BIDSLayout
from ieeg.io import raw_from_layout, bidspath_from_layout
from mne_bids import BIDSPath

try:
    from bids_writer import (
        EVENTS_COL_ORDER,
        EVENTS_JSON_METADATA,
        NPHONS,
        PHONEME_EVENTS_JSON_METADATA,
        phon_txt2df,
        remove_orphan_stimuli,
    )
except ImportError:
    from .bids_writer import (
        EVENTS_COL_ORDER,
        EVENTS_JSON_METADATA,
        NPHONS,
        PHONEME_EVENTS_JSON_METADATA,
        phon_txt2df,
        remove_orphan_stimuli,
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_DATA_DIR = Path(
    '~/Box/CoganLab/Data/Micro/BIDS_processing'
).expanduser()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def update_events_from_txt(
    bids_root,
    subject,
    task,
    stim_txt=None,
    resp_txt=None,
    stim_phon_txt=None,
    resp_phon_txt=None,
):
    """Update word-level and phoneme-level events.tsv from corrected .txt files.

    Parameters
    ----------
    bids_root : str or Path
        Root directory of the BIDS dataset.
    subject : str
        Subject identifier (without ``sub-`` prefix).
    task : str
        BIDS task label.
    stim_txt, resp_txt : str or Path, optional
        Paths to manually corrected word-level stimulus/response .txt files.
    stim_phon_txt, resp_phon_txt : str or Path, optional
        Paths to manually corrected phoneme-level stimulus/response .txt files.
    """
    has_word = stim_txt is not None or resp_txt is not None
    has_phon = stim_phon_txt is not None or resp_phon_txt is not None

    if not has_word and not has_phon:
        logger.info('No .txt files provided for updating events. Exiting.')
        return

    bids_layout = BIDSLayout(root=bids_root, derivatives=False)

    raw = raw_from_layout(
        bids_layout,
        subject=subject,
        task=task,
        extension='.edf',
        preload=False,
    )
    fs = raw.info['sfreq']

    bids_path = bidspath_from_layout(
        bids_layout,
        subject=subject,
        task=task,
        extension='.edf',
    )

    if has_word:
        _update_word_events(bids_path, subject, fs, stim_txt, resp_txt)

    if has_phon:
        _update_phoneme_events(
            bids_root, subject, task, fs, stim_phon_txt, resp_phon_txt,
        )


# ---------------------------------------------------------------------------
# Word-level helpers
# ---------------------------------------------------------------------------

def _update_word_events(bids_path, subject, fs, stim_txt, resp_txt):
    """Replace word-level events.tsv with data from corrected txt files."""
    events_tsv_path = bids_path.copy().update(
        suffix='events', extension='.tsv',
    )
    events_df = pd.read_csv(events_tsv_path.fpath, sep='\t')
    stim_events_df = events_df[events_df['trial_type'] == 'stimulus']
    resp_events_df = events_df[events_df['trial_type'] == 'response']

    if stim_txt:
        stim_events_df = _word_txt2df(stim_txt, 'stimulus', subject, fs)
    if resp_txt:
        resp_events_df = _word_txt2df(resp_txt, 'response', subject, fs)

    updated_df = (
        pd.concat([stim_events_df, resp_events_df], ignore_index=True)
        .sort_values('onset')
        .reset_index(drop=True)
    )

    updated_df = remove_orphan_stimuli(updated_df)

    # check that number of stimulus and response events are the same after removing orphan stimuli
    stim_events_updated = updated_df[updated_df['trial_type'] == 'stimulus']
    resp_events_updated = updated_df[updated_df['trial_type'] == 'response']
    if len(stim_events_updated) != len(resp_events_updated):
        logger.error('Number of stimulus and response events are not the same after removing orphan stimuli')
        return
    
    updated_df['trial'] = np.ceil(
        np.arange(1, len(updated_df) + 1) / 2
    ).astype(int)

    updated_df = updated_df[EVENTS_COL_ORDER]
    updated_df.to_csv(events_tsv_path.fpath, sep='\t', index=False)

    _write_json(
        events_tsv_path.fpath.with_suffix('.json'),
        EVENTS_JSON_METADATA,
    )
    logger.info('Updated word-level events.tsv at %s', events_tsv_path.fpath)


def _word_txt2df(txt_file, trial_type, subject, fs):
    """Convert a word-level MFA .txt file to a BIDS events DataFrame."""
    txt_file = _resolve_txt_path(txt_file, subject)
    event_df = pd.read_csv(
        txt_file, sep='\t', names=['onset', 'offset', 'value'],
    )
    event_df['duration'] = event_df['offset'] - event_df['onset']
    event_df['trial_type'] = trial_type
    event_df['sample'] = (event_df['onset'] * fs).astype(int)
    event_df['subject'] = subject
    event_df['trial'] = range(1, len(event_df) + 1)
    return event_df[EVENTS_COL_ORDER]


# ---------------------------------------------------------------------------
# Phoneme-level helpers
# ---------------------------------------------------------------------------

def _update_phoneme_events(
    bids_root, subject, task, fs, stim_phon_txt, resp_phon_txt,
):
    """Replace phoneme-level events.tsv in derivatives/phonemeLevel/."""
    bids_root = Path(bids_root)
    n_phons = NPHONS.get(task, 5)

    if stim_phon_txt is None or resp_phon_txt is None:
        logger.warning(
            'Both --stim_phon_txt and --resp_phon_txt are required for '
            'phoneme-level update. Skipping.',
        )
        return

    stim_phon_txt = _resolve_txt_path(stim_phon_txt, subject)
    resp_phon_txt = _resolve_txt_path(resp_phon_txt, subject)

    stim_df = phon_txt2df(
        stim_phon_txt, 'stimulus', subject, n_phons, fs, task,
    )
    resp_df = phon_txt2df(
        resp_phon_txt, 'response', subject, n_phons, fs, task,
    )

    events_df = (
        pd.concat([stim_df, resp_df])
        .sort_values('onset')
        .reset_index(drop=True)
    )
    events_df = remove_orphan_stimuli(events_df)
    
    # check that number of stimulus and response events are the same after removing orphan stimuli
    stim_events_updated = events_df[events_df['trial_type'] == 'stimulus']
    resp_events_updated = events_df[events_df['trial_type'] == 'response']
    if len(stim_events_updated) != len(resp_events_updated):
        logger.error('Number of stimulus and response events are not the same after removing orphan stimuli')
        return
    
    events_df['trial'] = np.ceil(
        np.arange(1, len(events_df) + 1) / (n_phons * 2)
    ).astype(int)

    events_path = BIDSPath(
        root=bids_root / 'derivatives' / 'phonemeLevel',
        subject=subject,
        task=task,
        datatype='ieeg',
        suffix='events',
        description='phonemeLevel',
        extension='.tsv',
        check=False,
    )
    events_path.mkdir(exist_ok=True)
    events_df.to_csv(events_path.fpath, sep='\t', index=False)

    _write_json(
        events_path.fpath.with_suffix('.json'),
        PHONEME_EVENTS_JSON_METADATA,
    )
    logger.info(
        'Updated phoneme-level events.tsv at %s', events_path.fpath,
    )


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _resolve_txt_path(txt_path, subject):
    """Resolve a txt file path, falling back to DEFAULT_DATA_DIR if needed."""
    txt_path = Path(txt_path)
    if txt_path.exists():
        return txt_path
    fallback = DEFAULT_DATA_DIR / f'sub-{subject}' / txt_path
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Text file not found at {txt_path} or {fallback}"
    )


def _write_json(json_path, metadata):
    """Write a JSON sidecar file."""
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    user_path = Path.home()
    default_bids_root = (
        user_path / 'Box' / 'CoganLab'
        / 'BIDS_1.0_Lexical_\u00b5ECoG' / 'BIDS'
    )
    default_task = 'lexical'

    parser = argparse.ArgumentParser(
        description=(
            'Update word-level and phoneme-level BIDS events.tsv '
            'from corrected .txt files.'
        ),
    )
    parser.add_argument(
        '--bids_root', default=default_bids_root,
        help='Path to BIDS root directory',
    )
    parser.add_argument(
        '--subject', required=True,
        help='Subject identifier',
    )
    parser.add_argument(
        '--task', default=default_task,
        help='Task identifier',
    )

    # Word-level txt files
    parser.add_argument(
        '--stim_txt', default='mfa_stim_words.txt',
        help='Path to corrected word-level stimulus .txt file',
    )
    parser.add_argument(
        '--resp_txt', default='mfa_adj_resp_words.txt',
        help='Path to corrected word-level response .txt file',
    )

    # Phoneme-level txt files (optional)
    parser.add_argument(
        '--stim_phon_txt', default='mfa_stim_phones.txt',
        help='Path to corrected phoneme-level stimulus .txt file',
    )
    parser.add_argument(
        '--resp_phon_txt', default='mfa_adj_resp_phones.txt',
        help='Path to corrected phoneme-level response .txt file',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    update_events_from_txt(
        bids_root=args.bids_root,
        subject=args.subject,
        task=args.task,
        stim_txt=args.stim_txt,
        resp_txt=args.resp_txt,
        stim_phon_txt=args.stim_phon_txt,
        resp_phon_txt=args.resp_phon_txt,
    )


if __name__ == '__main__':
    main()
