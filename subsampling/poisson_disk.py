"""Pitch (Poisson-disk) subsampling of high-density iEEG arrays.

Port of ``processing_utils/poisson_disk_sampling.py`` from
``cross_patient_speech_decoding`` adapted to the ``intraop_bids`` BIDS
channel-map convention (0-indexed, may contain NaN positions and
macro-electrode replicates).

Notable fixes vs. the reference implementation:

* The reference's ``pitch_subsample_sig_channels`` recursively retries
  with ``pitch_subsample_sig_channels(pt, nElec, data_path)`` -- it
  passes the electrode count where the pitch parameter is expected.
  Here the retry passes the original target pitch (and re-draws the
  Poisson sample with a fresh RNG state).
* MATLAB 1-indexed channel logic (``np.arange(1, maxElec + 1)``,
  ``-1`` offsets when indexing the chanmap, etc.) is removed; we use
  0-indexed channels throughout to match ``load_chanmap`` output.
* The nearest-neighbor search now uses ``scipy.spatial.cKDTree`` for
  O(n log n) behavior instead of the O(n^2) Python loop.
* Hardcoded per-subject mm dimensions are gone -- pitch metadata is
  resolved via ``chanmap_utils.resolve_pitch_for_subject``.
* Macro contacts repeated across many chanmap cells are de-duplicated
  so each physical contact is sampled at most once.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from . import chanmap_utils


def poisson_disk_sampling(
        domain: Tuple[int, int],
        spacing: float,
        n_points: int,
        threshold: int = 60,
        max_iter: int = 1000,
        show_iter: bool = False,
        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Bridson (2007) Poisson-disk sampling on a rectangular integer grid.

    Returns up to ``n_points`` floating-point sample coordinates that
    are pairwise at least ``spacing`` apart, suitable for being rounded
    to grid-cell indices afterwards.

    Parameters
    ----------
    domain : (int, int)
        Inclusive grid extent ``(rows, cols)``.  Returned coordinates
        will lie in ``[1, rows]`` x ``[1, cols]`` (the reference's
        1-indexed convention).  Callers should subtract 1 to convert
        to 0-indexed grid positions.
    spacing : float
        Minimum allowed pairwise distance between sampled points.
    n_points : int
        Target number of points.
    threshold : int
        Per-cell rejection threshold -- a cell is marked unavailable
        after this many failed dart throws.
    max_iter : int
        Hard cap on outer iterations to avoid infinite loops.
    show_iter : bool
        Print progress to stdout.
    rng : np.random.Generator or None
        RNG to draw from (defaults to ``np.random.default_rng()``).
    """
    if rng is None:
        rng = np.random.default_rng()

    ndim = len(domain)
    cell_size = spacing / np.sqrt(ndim)

    sgrid = [np.arange(1, s + 1, cell_size) for s in domain]
    sgrid = np.meshgrid(*sgrid, indexing='ij')
    size_grid = sgrid[0].shape

    sgrid = np.column_stack([g.ravel() for g in sgrid])
    empty_grid = np.ones(sgrid.shape[0], dtype=bool)
    n_empty_grid = int(np.sum(empty_grid))
    score_grid = np.zeros_like(empty_grid, dtype=int)

    pts: list[np.ndarray] = []
    pts_created = 0
    it = 0
    start = time.time()

    while pts_created < n_points and n_empty_grid > 0:
        if it > max_iter:
            print(f'[poisson_disk_sampling] reached max_iter with '
                  f'{pts_created}/{n_points} points; retrying with fresh RNG.')
            return poisson_disk_sampling(
                domain, spacing, n_points, threshold,
                max_iter=max_iter, show_iter=show_iter,
                rng=np.random.default_rng())

        avail = np.where(empty_grid)[0]
        n_dart = min(n_empty_grid, n_points)
        samp_cells = rng.choice(avail, size=n_dart, replace=False)
        tmp_pts = sgrid[samp_cells] + cell_size * rng.random((n_dart, ndim))

        if pts:
            all_pts = np.vstack((np.array(pts), tmp_pts))
        else:
            all_pts = tmp_pts

        # Distances to nearest *other* sampled point.
        tree = cKDTree(all_pts)
        # k=2: nearest is the point itself, second-nearest is the neighbor.
        nn_dist, _ = tree.query(tmp_pts, k=2)
        neigh_dist = nn_dist[:, 1]

        in_domain = np.all(tmp_pts < np.array(domain), axis=1)
        good_spacing = neigh_dist > spacing
        valid = in_domain & good_spacing

        score_pts = tmp_pts[~valid]
        tmp_pts = tmp_pts[valid]

        if tmp_pts.size:
            empty_idx = np.floor((tmp_pts + cell_size - 1) / cell_size
                                 ).astype(int)
            empty_lin = np.ravel_multi_index(empty_idx.T - 1, size_grid)
            empty_grid[empty_lin] = False

        if score_pts.size:
            score_idx = np.floor((score_pts + cell_size - 1) / cell_size
                                 ).astype(int)
            # Clip to size_grid bounds; failed darts can land just outside
            # the indexable range when the random offset exceeds cell_size.
            score_idx = np.clip(score_idx - 1, 0,
                                np.array(size_grid) - 1)
            score_lin = np.ravel_multi_index(score_idx.T, size_grid)
            np.add.at(score_grid, score_lin, 1)

        empty_grid = empty_grid & (score_grid < threshold)
        n_empty_grid = int(np.sum(empty_grid))
        pts.extend(tmp_pts)
        pts_created += int(tmp_pts.shape[0])
        it += 1

        if show_iter:
            elapsed = time.time() - start
            print(f'[poisson_disk_sampling] iter={it} pts={pts_created} '
                  f'empty={n_empty_grid} elapsed={elapsed:.3f}s')

    pts_arr = np.vstack(pts) if pts else np.zeros((0, ndim))
    if pts_arr.shape[0] > n_points:
        keep = rng.choice(pts_arr.shape[0], size=n_points, replace=False)
        pts_arr = pts_arr[keep]
    return pts_arr


def n_elec_for_pitch(grid_shape: Tuple[int, int],
                     base_pitch_mm: float,
                     target_pitch_mm: float) -> int:
    """Number of electrodes that fits the grid at the target pitch.

    Scales the total electrode count by ``(base_pitch / target_pitch)^2``
    -- i.e. doubling the pitch reduces electrode count by 4x.
    """
    if target_pitch_mm <= 0:
        raise ValueError(f"target_pitch_mm must be > 0; got {target_pitch_mm}")
    n_full = int(np.prod(grid_shape))
    return max(1, int(round(n_full * (base_pitch_mm / target_pitch_mm) ** 2)))


def pitch_subsample_channels(
        chanmap: np.ndarray,
        base_pitch_mm: float,
        target_pitch_mm: float,
        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return channel indices sampled at the target pitch.

    Implementation notes
    --------------------
    * The full grid spans ``grid_shape * base_pitch_mm`` so the target
      number of electrodes is derived from the ratio of pitches squared.
    * If the target density is at or above the full array, all channels
      are returned.
    * Otherwise, Poisson-disk sampling places ``n_elec`` points on the
      grid with minimum spacing tied to the target pitch; the points
      are rounded to grid cells and mapped to channel indices via the
      chanmap.  NaN cells and macro replicates are handled (each macro
      contact contributes at most once).
    """
    if rng is None:
        rng = np.random.default_rng()

    grid_shape = tuple(chanmap.shape)
    n_full = chanmap_utils.unique_channel_count(chanmap)
    n_elec = n_elec_for_pitch(grid_shape, base_pitch_mm, target_pitch_mm)

    all_ch = chanmap_utils.all_channels(chanmap)
    if n_elec >= n_full:
        return all_ch

    spacing = float(np.floor(np.sqrt(np.prod(grid_shape) / n_elec)))
    spacing = max(spacing, 1.0)

    pts = poisson_disk_sampling(grid_shape, spacing, n_elec, rng=rng)
    # Reference uses 1-indexed grid coordinates; subtract 1 for 0-indexed.
    rc_idx = np.round(pts).astype(int) - 1
    rc_idx = np.clip(rc_idx, 0,
                     np.array(grid_shape) - 1) if rc_idx.size else rc_idx

    elec_pt = chanmap_utils.grid_idx_to_channels(chanmap, rc_idx)

    if elec_pt.size < n_elec and spacing == 1.0:
        # Reference fallback: top up with uniform draws from the
        # unsampled channels when minimum spacing has bottomed out.
        remaining = np.setdiff1d(all_ch, elec_pt, assume_unique=False)
        n_extra = min(remaining.size, n_elec - int(elec_pt.size))
        if n_extra > 0:
            extra = rng.choice(remaining, size=n_extra, replace=False)
            elec_pt = np.concatenate((elec_pt, extra))

    return np.asarray(elec_pt, dtype=int)


def pitch_subsample_picks(
        ch_names_all,
        chanmap: np.ndarray,
        base_pitch_mm: float,
        target_pitch_mm: float,
        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Wrapper around ``pitch_subsample_channels`` that validates picks
    against an ``epochs.ch_names`` list."""
    ch_idx = pitch_subsample_channels(
        chanmap, base_pitch_mm, target_pitch_mm, rng=rng)
    return chanmap_utils.channels_to_picks(ch_names_all, ch_idx)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    grid_x, grid_y = 8, 16
    for n_elec in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        print(f'### Sampling for n_elec = {n_elec} ###')
        spacing = float(np.floor(np.sqrt(grid_x * grid_y / n_elec)))
        spacing = max(spacing, 1.0)

        n_grids = 3
        fig, axes = plt.subplots(1, n_grids, figsize=(12, 4))
        for i in range(n_grids):
            pts = poisson_disk_sampling((grid_x, grid_y), spacing, n_elec)
            pts_idx = np.round(pts).astype(int) - 1
            print(f'  drew {pts_idx.shape[0]} points')
            grid = np.zeros((grid_x, grid_y))
            grid[pts_idx[:, 0], pts_idx[:, 1]] = 1
            axes[i].imshow(grid.T, cmap='gray', origin='lower')
            axes[i].set_title(f'iter {i + 1}')
        fig.suptitle(f'n_elec={n_elec}')
        plt.tight_layout()
        plt.show()
