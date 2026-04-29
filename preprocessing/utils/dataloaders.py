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
        scope='raw',
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


def load_derivative_raw(bids_layout, subject, derivative_name='phonemeLevel',
                        desc='phonemeLevel', **kwargs):
    """Load a raw recording from a BIDS derivatives subdirectory.

    Parameters
    ----------
    bids_layout : BIDSLayout
        The BIDS layout object, initialized with ``derivatives=True``.
    subject : str
        The subject identifier (without 'sub-' prefix).
    derivative_name : str
        Name of the derivative dataset under ``derivatives/``.
    desc : str
        The ``desc`` entity used in derivative filenames.
    **kwargs
        Additional keyword arguments passed to ``raw_from_layout``.
    """
    deriv_key = f'derivatives/{derivative_name}'
    deriv_layout = bids_layout.derivatives[deriv_key]
    raw = raw_from_layout(
        deriv_layout, subject=subject, extension='.edf',
        desc=desc, **kwargs
    )
    if not raw:
        raise ValueError(
            f'No derivative raw for {subject} in {derivative_name}')

    raw.load_data()
    raw._data *= 1e6  # convert to uV

    raw.drop_channels(['Trigger']) if 'Trigger' in raw.ch_names else None

    return raw


def load_channels_tsv(bids_layout, subject):
    """Load a subject's channels.tsv as a DataFrame.

    Parameters
    ----------
    bids_layout : BIDSLayout
        The BIDS layout object for the dataset.
    subject : str
        The subject identifier (without 'sub-' prefix).

    Returns
    -------
    pd.DataFrame
        The channels.tsv contents.
    """
    bids_root = Path(bids_layout.root)
    channels_files = list(bids_root.glob(
        f'sub-{subject}/**/sub-{subject}*_channels.tsv'))
    if not channels_files:
        raise FileNotFoundError(
            f'No channels.tsv found for subject {subject}')

    return pd.read_csv(channels_files[0], sep='\t')


def load_chanmap(bids_layout, subject):
    """Build a channel map grid from the channels.tsv file.

    Macro electrodes are expanded via flood fill so their channel
    index occupies all adjacent NaN cells.

    Parameters
    ----------
    bids_layout : BIDSLayout
        The BIDS layout object for the dataset.
    subject : str
        The subject identifier (without 'sub-' prefix).

    Returns
    -------
    np.ndarray
        2D array where each cell contains a channel index, NaN for
        empty positions, or a duplicated index for macro electrode
        spans.
    """
    channels_df = load_channels_tsv(bids_layout, subject)
    chan_coords = _get_chan_coords(channels_df['name'].tolist())

    n_rows = np.max(chan_coords[:, 0]) + 1
    n_cols = np.max(chan_coords[:, 1]) + 1

    chanmap = np.full((n_rows, n_cols), np.nan)
    for i, (row, col) in enumerate(chan_coords):
        chanmap[row, col] = i

    if 'electrode_type' in channels_df.columns:
        for i, etype in enumerate(channels_df['electrode_type']):
            if etype == 'macro':
                r, c = chan_coords[i]
                visited = _flood_fill_nans(chanmap, r, c)
                for nr, nc in visited:
                    if (nr, nc) != (r, c):
                        chanmap[nr, nc] = chanmap[r, c]

    return chanmap


def load_electrode_types(bids_layout, subject):
    """Load electrode types from channels.tsv.

    Parameters
    ----------
    bids_layout : BIDSLayout
        The BIDS layout object for the dataset.
    subject : str
        The subject identifier (without 'sub-' prefix).

    Returns
    -------
    dict or None
        Mapping from channel name to electrode type string,
        or None if the electrode_type column is not present.
    """
    channels_df = load_channels_tsv(bids_layout, subject)
    if 'electrode_type' not in channels_df.columns:
        return None
    return dict(zip(channels_df['name'], channels_df['electrode_type']))


def _get_chan_coords(chan_names):
    """Channel names are in the format: 'RX-CY' where X is the row and Y is the column"""
    coords = np.array([[int(name.split('-')[0][1:]), int(name.split('-')[1][1:])] for name in chan_names])

    return coords


def _flood_fill_nans(chan_map, start_r, start_c):
    """Find all NaN cells reachable from a position via adjacent NaN cells.

    Used to determine the physical extent of a macro electrode in the grid.
    """
    n_rows, n_cols = chan_map.shape
    visited = {(start_r, start_c)}
    queue = [(start_r, start_c)]

    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue
            if 0 <= nr < n_rows and 0 <= nc < n_cols and np.isnan(
                    chan_map[nr, nc]):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return visited


def get_neighbors(chan_map, r, c):
    """Get the neighbors of a channel in the channel map.

    Parameters
    ----------
    chan_map : np.ndarray
        The channel map.
    r : int
    c : int
    """
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n_rows and 0 <= nc < n_cols and not np.isnan(chan_map[nr, nc]):
            neighbors.append((nr, nc))
    return neighbors