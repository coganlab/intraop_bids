"""
Updates the events.tsv in a BIDS dataset based on a provided .txt file.

1) Load subj raw from BIDS
2) Read events.tsv annotations
3) Determine which events (stim/resp) needed to be updated from CLI args
4) Load .txt files for stim and/or resp events provided by CLI args
5) Update events.tsv accordingly
6) Save updated events.tsv back to BIDS
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from bids import BIDSLayout
from ieeg.io import raw_from_layout, bidspath_from_layout
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def update_events_from_txt(bids_root, subject, task,
                           stim_txt=None, resp_txt=None):
    """Update events.tsv in BIDS dataset from provided .txt files."""
    if stim_txt is None and resp_txt is None:
        logger.info('No .txt files provided for updating events. Exiting.')
        return

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

    bids_path = bidspath_from_layout(
        bids_layout,
        subject=subject,
        task=task,
        extension='.edf',
    )

    # Load existing events.tsv
    events_tsv_path = bids_path.copy().update(suffix='events',
                                              extension='.tsv')
    events_df = pd.read_csv(events_tsv_path, sep='\t')
    stim_events_df = events_df[events_df['trial_type'] == 'stimulus']
    resp_events_df = events_df[events_df['trial_type'] == 'response']

    # Update stim events if stim_txt is provided
    if stim_txt:
        stim_events_df = txt2df(stim_txt, 'stimulus', subject,
                                raw.info['sfreq'])
        print(stim_events_df.head())

    # Update resp events if resp_txt is provided
    if resp_txt:
        resp_events_df = txt2df(resp_txt, 'response', subject,
                                raw.info['sfreq'])
        print(resp_events_df.head())

    # Combine updated events
    updated_events_df = pd.concat([stim_events_df, resp_events_df],
                                  ignore_index=True)
    updated_events_df = updated_events_df.sort_values(by='onset')

    # Remove any stimulus events without associated response events
    for idx, row in updated_events_df.iterrows():
        if row['trial_type'] == 'stimulus':
            stim_onset = row['onset']
            resp_events = updated_events_df[
                (updated_events_df['trial_type'] == 'response') &
                (updated_events_df['onset'] > stim_onset) &
                (updated_events_df['onset'] <= stim_onset + 3.0)
            ]
            if resp_events.empty:
                logger.info(f'Removing stimulus event at onset {stim_onset} '
                            'without associated response event.')
                updated_events_df = updated_events_df.drop(idx)
    
    # Re-index trials
    updated_events_df = updated_events_df.reset_index(drop=True)
    updated_events_df['trial'] = np.ceil(np.arange(1, len(updated_events_df) + 1) / 2).astype(int)
    

    # Save updated events.tsv
    updated_events_df.to_csv(events_tsv_path, sep='\t', index=False)
    logger.info(f'Updated events.tsv saved to {events_tsv_path}')


def txt2df(txt_file, trial_type, subject, fs):
    event_df = pd.read_csv(txt_file, sep='\t', names=['onset', 'offset',
                                                      'value'])
    event_df['duration'] = event_df['offset'] - event_df['onset']
    event_df['trial_type'] = trial_type
    event_df['sample'] = (event_df['onset'] * fs).astype(int)
    event_df['subject'] = subject
    event_df['trial'] = range(1, len(event_df) + 1)
    event_df = event_df[['subject', 'trial', 'onset', 'duration',
                         'value', 'trial_type', 'sample']]
    return event_df


def parse_args():
    user_path = Path.home()
    default_bids_root = (user_path / 'Box' / 'CoganLab' /
                         'BIDS_1.0_Phoneme_Sequence_uECoG' / 'BIDS')
    default_task = 'phoneme'
    parser = argparse.ArgumentParser(
            description="Update BIDS events.tsv from provided .txt files."
        )
    parser.add_argument('--bids_root', default=default_bids_root,
                        help='Path to BIDS root directory')
    parser.add_argument('--subject', required=True, help='Subject '
                        'identifier')
    parser.add_argument('--task', default=default_task, help='Task identifier')
    parser.add_argument('--stim_txt', help='Path to stim events .txt file')
    parser.add_argument('--resp_txt', help='Path to resp events .txt file')

    return parser.parse_args()


def main():
    args = parse_args()

    update_events_from_txt(
        bids_root=args.bids_root,
        subject=args.subject,
        task=args.task,
        stim_txt=args.stim_txt,
        resp_txt=args.resp_txt
    )


if __name__ == '__main__':
    main()
