import numpy as np
import pandas as pd
from pathlib import Path
from mne_bids import BIDSPath
from ieeg.io import raw_from_layout


def load_raw(bids_layout, subject, **kwargs):
    # search query
    raw = raw_from_layout(
        bids_layout,
        subject=subject,
        extension='.edf',
        # desc=None,
        **kwargs
    )

    # check the raw data was found with query
    if not raw:
        raise ValueError(f'No raw data found for subject: {subject}')

    raw.load_data()
    raw._data *= 1e6  # convert to uV

    # remove Trigger channel we have any
    raw.drop_channels(['Trigger']) if 'Trigger' in raw.ch_names else None

    return raw


def load_chanmap(subject, bids_layout, **kwargs):
    bids_root = Path(bids_layout.root)

    bids_path = BIDSPath(
        root=bids_root / 'derivatives' / 'channelMap',
        subject=subject,
        datatype='channelMap',
        extension='.tsv',
        suffix='channelMap',
        check=False,
        **kwargs
    )
    chanmap_df = pd.read_csv(bids_path.match()[0], sep='\t')

    n_rows = np.max(chanmap_df['row'].values) + 1
    n_cols = np.max(chanmap_df['col'].values) + 1

    chanmap = np.full((n_rows, n_cols), np.nan)
    for _, row in chanmap_df.iterrows():
        chanmap[row['row'], row['col']] = int(row['name'])

    return chanmap
