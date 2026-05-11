"""Spatial averaging of contacts to simulate larger contact sizes.

Port of ``processing_utils/spatial_avg_subsampling.py`` from
``cross_patient_speech_decoding``, restructured so that:

* The averaging works on *unprocessed* raw time-domain data
  ``(n_channels, n_times)`` rather than already-epoched ``(trials,
  chans_x, chans_y, time)`` arrays.  The intent is to faithfully
  simulate physically larger contacts: spatially average first, then
  let the standard preprocessing pipeline (outlier detection, notch,
  CAR) treat the averaged signal as if it came from a real macro array.
* ``np.nanmean`` is used to be robust to missing channels even though
  raw recordings carry no NaNs.
* Windows are square ``contact_size x contact_size`` and are centered
  in the grid via ``shift = (grid_dim % contact_size) // 2``.  When the
  leftover padding along a dimension is odd, the extra cell sits at the
  bottom / right edge (this matches the reference behavior).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import chanmap_utils
from .coverage import grid_subsample_idxs


def spatial_avg_idxs(grid_shape: Tuple[int, int],
                     contact_size: int) -> List[np.ndarray]:
    """Return the grid-cell groupings for ``contact_size`` averaging.

    Square non-overlapping windows of size ``contact_size`` are tiled
    across the grid and centered: the leftover padding ``grid_dim %
    contact_size`` is split as evenly as possible, with the extra cell
    (when the leftover is odd) appearing on the bottom / right edge.
    """
    if contact_size <= 0:
        raise ValueError(
            f"contact_size must be positive; got {contact_size}")
    win = (contact_size, contact_size)
    step = (contact_size, contact_size)
    shift_x = (grid_shape[0] % contact_size) // 2
    shift_y = (grid_shape[1] % contact_size) // 2
    start = (shift_x, shift_y)
    return grid_subsample_idxs(grid_shape, win, step=step, start=start)


def average_group_indices(
        chanmap: np.ndarray,
        contact_size: int,
        min_valid_frac: float = 0.5
        ) -> Tuple[List[np.ndarray], List[Tuple[int, int]],
                   List[Tuple[int, int]]]:
    """Return one channel-index group per averaged contact.

    Parameters
    ----------
    chanmap : np.ndarray
        Output of ``load_chanmap``.
    contact_size : int
        Edge length of the square averaging window.
    min_valid_frac : float
        Skip windows where at least ``(1 - min_valid_frac)`` of the cells
        are NaN.  Defaults to 0.5 (same threshold as the reference).

    Returns
    -------
    groups : list of np.ndarray
        Each entry is the (unique) channel indices to be averaged for
        one virtual contact.
    reduced_rc : list of (int, int)
        ``(row, col)`` position of each averaged contact on the *reduced*
        grid (one cell per virtual contact, used for naming like
        ``R{r}-C{c}`` in the derivative channels.tsv).
    centroids : list of (int, int)
        ``(row, col)`` centroid of each averaging window on the
        *original* grid (saved for documentation in the derivative
        channel-map tsv).
    """
    groups: List[np.ndarray] = []
    reduced_rc: List[Tuple[int, int]] = []
    centroids: List[Tuple[int, int]] = []

    if contact_size == 1:
        # Each grid cell stays a single channel; preserve grid order.
        for r in range(chanmap.shape[0]):
            for c in range(chanmap.shape[1]):
                if np.isnan(chanmap[r, c]):
                    continue
                groups.append(np.array([int(chanmap[r, c])], dtype=int))
                reduced_rc.append((r, c))
                centroids.append((r, c))
        return groups, reduced_rc, centroids

    # Walk the centered tiling row-by-row so the resulting (i, j) index
    # mirrors the spatial layout of the reduced grid.
    shift_x = (chanmap.shape[0] % contact_size) // 2
    shift_y = (chanmap.shape[1] % contact_size) // 2
    n_rows = (chanmap.shape[0] - shift_x) // contact_size
    n_cols = (chanmap.shape[1] - shift_y) // contact_size

    for i in range(n_rows):
        r0 = shift_x + i * contact_size
        for j in range(n_cols):
            c0 = shift_y + j * contact_size
            rs = np.arange(r0, r0 + contact_size)
            cs = np.arange(c0, c0 + contact_size)
            cells = np.array(np.meshgrid(rs, cs, indexing='ij')
                             ).reshape(2, -1).T
            vals = chanmap[cells[:, 0], cells[:, 1]]
            nan_mask = np.isnan(vals)
            if nan_mask.sum() >= len(vals) * (1 - min_valid_frac):
                continue
            good = ~nan_mask
            ch_idx = vals[good].astype(int)
            # Deduplicate macro repeats so each physical contact
            # contributes once to the spatial average.
            _, first_idx = np.unique(ch_idx, return_index=True)
            ch_idx = ch_idx[np.sort(first_idx)]
            if ch_idx.size == 0:
                continue
            groups.append(ch_idx)
            reduced_rc.append((i, j))
            cr = int(r0 + contact_size // 2)
            cc = int(c0 + contact_size // 2)
            centroids.append((cr, cc))
    return groups, reduced_rc, centroids


def spatial_average_data(data: np.ndarray,
                         groups: Sequence[np.ndarray]) -> np.ndarray:
    """Average over channel groups along axis ``-2`` of ``data``.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(..., n_channels, n_times)``.  Works for both raw
        ``(n_channels, n_times)`` and epoched ``(n_trials, n_channels,
        n_times)`` arrays.
    groups : sequence of np.ndarray
        Each entry lists the channel indices to average together.

    Returns
    -------
    np.ndarray
        Same leading shape with ``n_channels`` replaced by ``len(groups)``.
    """
    out_shape = list(data.shape)
    out_shape[-2] = len(groups)
    out = np.empty(out_shape, dtype=data.dtype)
    for i, idx in enumerate(groups):
        # nanmean handles defensively-missing channels; raw recordings
        # generally have no NaNs.
        out[..., i, :] = np.nanmean(data[..., idx, :], axis=-2)
    return out


def averaged_channel_names(reduced_rc: Sequence[Tuple[int, int]]
                           ) -> List[str]:
    """Construct ``R{i}-C{j}`` names from reduced-grid coordinates.

    Channel names use the standard ``RX-CY`` format so the existing
    ``load_chanmap`` parser works on the derivative without changes.
    """
    return [f'R{i}-C{j}' for i, j in reduced_rc]


def averaged_chanmap(reduced_rc: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Build a chanmap for the averaged channels indexed in reduced-grid
    order (one cell per virtual contact)."""
    if not reduced_rc:
        return np.zeros((0, 0))
    rows = [r for r, _ in reduced_rc]
    cols = [c for _, c in reduced_rc]
    n_rows = max(rows) + 1
    n_cols = max(cols) + 1
    cmap = np.full((n_rows, n_cols), np.nan)
    for i, (r, c) in enumerate(reduced_rc):
        cmap[r, c] = i
    return cmap


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for grid_shape in [(8, 16), (10, 16), (10, 17)]:
        for contact_size in [2, 3, 4, 8]:
            cells = spatial_avg_idxs(grid_shape, contact_size)
            grid = np.zeros(grid_shape)
            for k, rc in enumerate(cells, start=1):
                grid[rc[:, 0], rc[:, 1]] = k
            shift = ((grid_shape[0] % contact_size) // 2,
                     (grid_shape[1] % contact_size) // 2)
            print(f'grid={grid_shape} contact={contact_size} '
                  f'n_groups={len(cells)} shift={shift}')
            plt.figure()
            plt.imshow(grid.T, cmap='tab20', origin='lower')
            plt.title(f'{grid_shape} cs={contact_size} shift={shift}')
            plt.show()
