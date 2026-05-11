"""Coverage (subarray) subsampling of high-density iEEG arrays.

Port of ``processing_utils/grid_subsampling.py`` from
``cross_patient_speech_decoding``, adapted to the ``intraop_bids`` BIDS
channel-map convention.

Differences from the reference:

* Typo ``grid_susbsample_idxs`` is corrected to ``grid_subsample_idxs``.
* The reference's hardcoded NaN-edge trimming (``chanMap.shape[0] == 24``
  / ``shape[1] == 24``) and conditional ``winSize`` transposition were
  artifacts of the legacy ``.mat`` channel-map format; ``load_chanmap``
  in ``intraop_bids`` already returns a clean grid, so these branches
  are dropped.
* Channel indices are 0-indexed.  NaN cells and macro replicates are
  handled via ``chanmap_utils.grid_idx_to_channels``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from . import chanmap_utils


def grid_subsample_idxs(
        grid_shape: Tuple[int, int],
        win_size: Tuple[int, int],
        step: Tuple[int, int] = (1, 1),
        start: Tuple[int, int] = (0, 0)) -> List[np.ndarray]:
    """Enumerate all valid ``win_size`` subgrid placements on a grid.

    Parameters
    ----------
    grid_shape : (int, int)
    win_size : (int, int)
    step : (int, int)
        Step between consecutive subgrid origins along each axis.
    start : (int, int)
        First subgrid origin (defaults to top-left).

    Returns
    -------
    list of np.ndarray
        Each entry is an ``(n_cells, 2)`` array of ``(row, col)`` cells
        covered by one subgrid placement.
    """
    start_x = np.arange(start[0], grid_shape[0] - win_size[0] + 1, step[0])
    start_y = np.arange(start[1], grid_shape[1] - win_size[1] + 1, step[1])

    starts = np.array(np.meshgrid(start_x, start_y, indexing='ij'))
    starts = starts.reshape(2, -1).T

    out = []
    for x, y in starts:
        rs = np.arange(x, x + win_size[0])
        cs = np.arange(y, y + win_size[1])
        rc = np.array(np.meshgrid(rs, cs, indexing='ij')).reshape(2, -1).T
        out.append(rc)
    return out


def coverage_subgrids(chanmap: np.ndarray,
                      win_size: Sequence[int]) -> List[np.ndarray]:
    """List of channel-index arrays, one per valid subgrid placement."""
    win = (int(win_size[0]), int(win_size[1]))
    grid_idxs = grid_subsample_idxs(chanmap.shape, win)
    out = []
    for rc in grid_idxs:
        ch_idx = chanmap_utils.grid_idx_to_channels(chanmap, rc)
        if ch_idx.size > 0:
            out.append(ch_idx)
    return out


def coverage_subsample_picks(
        ch_names_all: Sequence[str],
        chanmap: np.ndarray,
        win_size: Sequence[int]) -> List[np.ndarray]:
    """Same as ``coverage_subgrids`` but validates picks against
    ``epochs.ch_names``."""
    return [chanmap_utils.channels_to_picks(ch_names_all, ch_idx)
            for ch_idx in coverage_subgrids(chanmap, win_size)]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    grid_shape = (8, 16)
    win_size = (6, 12)
    grids = grid_subsample_idxs(grid_shape, win_size)
    print(f'Got {len(grids)} possible grids')

    for rc in grids:
        grid = np.zeros(grid_shape)
        grid[rc[:, 0], rc[:, 1]] = 1
        plt.imshow(grid.T, cmap='gray', origin='lower', clim=[0, 1])
        plt.show()
