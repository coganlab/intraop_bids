""" Pairs pre-formatted raw BIDS data with phoneme-level label annotations
input from text files.

Steps:
1) Loads BIDS data for a given subject/task
2) Reads phoneme-level annotations from provided .txt files
3) Creates a new events.tsv file with phoneme-level events
4) Saves the new events.tsv back to the BIDS dataset
"""

import argparse
import logging
import re
from pathlib import Path
from bids import BIDSLayout
from mne_bids import BIDSPath
from ieeg.io import raw_from_layout, bidspath_from_layout, save_derivative
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PS2ARPA = {'a': 'AA', 'ae': 'EH', 'i': 'IY', 'u': 'UW', 'b': 'B', 'p': 'P',
           'v': 'V', 'g': 'G', 'k': 'K', 'UH': 'UW', 'AE': 'EH'}


def load_raw(bids_root, subject):
    """Load raw data from BIDS dataset."""
    bids_layout = BIDSLayout(
        root=bids_root,
        derivatives=False,
    )

    raw = raw_from_layout(
        bids_layout,
        subject=subject,
        extension='.edf',
        preload=False,
    )
    return raw, bids_layout


def create_phon_tsv(stim_txt, resp_txt, subject, nPhons, fs):
    """Create phoneme-level annotations from .txt files."""
    stim_df = txt2df(stim_txt, 'stimulus', subject, nPhons, fs)
    resp_df = txt2df(resp_txt, 'response', subject, nPhons, fs)

    tot_df = pd.concat([stim_df, resp_df]).sort_values(by='onset').reset_index(drop=True)
    print(tot_df.head(n=10))

    # Remove any stimulus events without associated response events
    for idx, row in tot_df.iterrows():
        if 'stimulus' in row['trial_type']:
            stim_onset = row['onset']
            resp_events = tot_df[
                (tot_df['trial_type'].apply(lambda x: x.split('/')[0]) == 'response') &
                (tot_df['onset'] > stim_onset) &
                (tot_df['onset'] <= stim_onset + 3.0)
            ]
            if resp_events.empty:
                logger.info(f'Removing stimulus event at onset {stim_onset} '
                            'without associated response event.')
                tot_df = tot_df.drop(idx)

    # Re-index trials
    tot_df = tot_df.reset_index(drop=True)
    tot_df['trial'] = np.ceil(np.arange(1, len(tot_df) + 1) / (nPhons*2)).astype(int)

    return tot_df


def txt2df(txt_path, trial_type, subject, nPhons, fs):
    """Convert .txt file to events DataFrame."""
    df = pd.read_csv(txt_path, sep='\t', names=['onset', 'offset', 'phoneme'])
    df['duration'] = df['offset'] - df['onset']
    df['value'] = df['phoneme'].apply(lambda x: PS2ARPA.get(remove_arpabet_stress(x), remove_arpabet_stress(x)))
    df['subject'] = subject
    df['sample'] = (df['onset'] * fs).astype(int)
    df['trial'] = (np.arange(len(df)) // nPhons) + 1
    df['trial_type'] = (np.arange(len(df)) % nPhons) + 1
    df['trial_type'] = df['trial_type'].apply(lambda x: f'{trial_type}/{x}')
    df = df[['subject', 'trial', 'onset', 'duration', 'value', 'trial_type',
             'sample']]
    return df


def save_events_tsv(bids_root, subject, task, events_df):
    """Save events DataFrame to events.tsv in BIDS dataset."""
    events_path = BIDSPath(
        root=bids_root / 'derivatives' / 'phonemeLevel',
        subject=subject,
        task=task,
        acquisition='01',
        run='01',
        datatype='ieeg',
        suffix='events',
        description='phonemeLevel',
        extension='.tsv',
        check=False
    )

    events_df.to_csv(events_path.fpath, sep='\t', index=False)
    logger.info(f'Saved events.tsv to {events_path.fpath}')


def remove_arpabet_stress(phoneme: str) -> str:
    """Remove stress markers from ARPAbet phonemes.

    E.g., 'AH0' -> 'AH', 'IY1' -> 'IY'
    """
    return re.sub(r'\d', '', phoneme)


def parse_args():
    user_path = Path.home()
    default_bids_root = (user_path / 'Box' / 'CoganLab' /
                         'BIDS_1.0_Phoneme_Sequence_uECoG' / 'BIDS')
    defatult_task = 'phoneme'

    parser = argparse.ArgumentParser(
        description='Create phoneme-level events.tsv from .txt annotations.'
    )
    parser.add_argument('--bids_root', default=default_bids_root,
                        help='Path to the BIDS root directory.')
    parser.add_argument('--subject', type=str, required=True,
                        help='Subject identifier (e.g., sub-01).')
    parser.add_argument('--task', type=str, default=defatult_task,
                        help='Task identifier (e.g., task-speech).')
    parser.add_argument('--stim_txt', required=True, type=str,
                        help='Path to stimulus .txt annotation file.')
    parser.add_argument('--resp_txt', required=True, type=str,
                        help='Path to response .txt annotation file.')
    return parser.parse_args()


def main():
    args = parse_args()
    bids_root = Path(args.bids_root)

    if args.task == 'phoneme':
        nPhons = 3
    elif args.task == 'lexical':
        nPhons = 5
    else:
        raise ValueError(f'Unknown task: {args.task}')

    raw, _ = load_raw(
        bids_root=bids_root,
        subject=args.subject,
    )

    events_df = create_phon_tsv(
        stim_txt=args.stim_txt,
        resp_txt=args.resp_txt,
        subject=args.subject,
        nPhons=nPhons,
        fs=raw.info['sfreq']
    )

    # save raw data to phonemeLevel derivative folder
    deriv_layout = BIDSLayout(
        root=bids_root,
        derivatives=True,
    )

    save_derivative(raw, deriv_layout, 'phonemeLevel', True)

    save_events_tsv(
        bids_root=bids_root,
        subject=args.subject,
        task=args.task,
        events_df=events_df
    )


if __name__ == '__main__':
    main()
