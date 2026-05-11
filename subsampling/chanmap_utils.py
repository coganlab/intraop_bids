"""Helpers for working with the channel-map grid returned by
``preprocessing.utils.dataloaders.load_chanmap``.

The chanmap has shape ``(n_rows, n_cols)`` with float entries:
- ``NaN`` for empty grid positions,
- the (0-indexed) channel index into ``channels.tsv`` / ``epochs.ch_names``
  for occupied positions,
- the same channel index repeated across all cells that belong to a
  flood-filled macro contact.

Subsampling routines need to translate between grid coordinates and the
flat channel indices used by MNE epoch arrays, while gracefully handling
the macro-electrode and NaN edge cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


def clean_chanmap(chanmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return a copy of ``chanmap`` with macro-electrode duplicates marked.

    The input chanmap (see ``load_chanmap``) may contain the same channel
    index in many adjacent cells when an electrode is flagged as a macro
    contact.  For pitch / coverage subsampling we want to treat each
    physical contact as a single sampling target, so we return a boolean
    ``valid_mask`` that is ``True`` only at the canonical (first-seen)
    cell of each channel.

    Parameters
    ----------
    chanmap : np.ndarray
        Float 2D channel map.

    Returns
    -------
    grid : np.ndarray
        Same data as ``chanmap`` (a copy).
    valid_mask : np.ndarray of bool, same shape as ``chanmap``
        ``True`` at exactly one cell per unique non-NaN channel index.
    """
    grid = np.array(chanmap, copy=True)
    valid_mask = np.zeros(grid.shape, dtype=bool)
    seen = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            val = grid[r, c]
            if np.isnan(val):
                continue
            key = int(val)
            if key in seen:
                continue
            seen.add(key)
            valid_mask[r, c] = True
    return grid, valid_mask


def grid_idx_to_channels(chanmap: np.ndarray,
                         rc_idxs: np.ndarray) -> np.ndarray:
    """Map a list of ``(row, col)`` grid coordinates to channel indices.

    Parameters
    ----------
    chanmap : np.ndarray
        Output of ``load_chanmap`` (may contain NaN / repeated values).
    rc_idxs : np.ndarray of shape (n, 2)
        Grid coordinates to look up.

    Returns
    -------
    np.ndarray of int
        Unique channel indices (NaNs dropped, duplicates removed,
        order preserved by first occurrence).
    """
    if rc_idxs.size == 0:
        return np.array([], dtype=int)
    vals = chanmap[rc_idxs[:, 0], rc_idxs[:, 1]]
    good = ~np.isnan(vals)
    vals = vals[good].astype(int)
    # Preserve first-occurrence order while de-duplicating macro repeats.
    _, unique_idx = np.unique(vals, return_index=True)
    return vals[np.sort(unique_idx)]


def channels_to_picks(ch_names_all: Sequence[str],
                      ch_indices: Iterable[int]) -> np.ndarray:
    """Validate channel indices against ``ch_names_all``.

    For an MNE epochs object loaded from BIDS, ``epochs.ch_names`` is
    ordered the same as ``channels.tsv``, which is what ``load_chanmap``
    indexes against.  This helper simply asserts the indices are in
    range and returns them as ``int``.
    """
    ch_indices = np.asarray(list(ch_indices), dtype=int)
    if ch_indices.size and (ch_indices.min() < 0
                            or ch_indices.max() >= len(ch_names_all)):
        raise ValueError(
            f"Channel index out of range [0, {len(ch_names_all)}): "
            f"got [{ch_indices.min()}, {ch_indices.max()}]")
    return ch_indices


def chanmap_idxs_to_epoch_picks(chanmap_ch_idxs: Iterable[int],
                                full_ch_names: Sequence[str],
                                epoch_ch_names: Sequence[str]
                                ) -> np.ndarray:
    """Translate chanmap channel indices to picks into ``epoch_ch_names``.

    ``chanmap_ch_idxs`` are indices into ``full_ch_names`` (the original
    ``channels.tsv`` order, which is what ``load_chanmap`` returns).
    Channels not present in ``epoch_ch_names`` (e.g. sig-channel
    selection has filtered them out) are silently dropped.
    """
    ep_index = {n: i for i, n in enumerate(epoch_ch_names)}
    picks = []
    for idx in chanmap_ch_idxs:
        name = full_ch_names[int(idx)]
        if name in ep_index:
            picks.append(ep_index[name])
    return np.asarray(picks, dtype=int)


def _config_dir() -> Path:
    return Path(__file__).resolve().parent / 'config'


def load_array_pitch(array_type: str,
                     config_path: Optional[Path] = None) -> dict:
    """Load base pitch / grid info for an electrode array.

    Parameters
    ----------
    array_type : str
        Key in ``subsampling/config/array_pitch.yaml``.
    config_path : Path or None
        Override the default config file location.

    Returns
    -------
    dict
        Keys include ``pitch_mm`` (float) and optionally ``grid_dims``
        (``[n_rows, n_cols]``).
    """
    if config_path is None:
        config_path = _config_dir() / 'array_pitch.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if array_type not in cfg:
        raise KeyError(
            f"Unknown array_type '{array_type}' in {config_path}. "
            f"Known: {sorted(cfg.keys())}")
    return cfg[array_type]


def lookup_subject_array(subject: str,
                         config_path: Optional[Path] = None) -> str:
    """Look up the array_type for a subject.

    Parameters
    ----------
    subject : str
        Subject ID without ``sub-`` prefix (e.g. ``'S41'``).
    config_path : Path or None
        Override the default ``subject_array.yaml`` path.
    """
    if config_path is None:
        config_path = _config_dir() / 'subject_array.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if subject not in cfg:
        raise KeyError(
            f"Subject '{subject}' not in {config_path}. "
            f"Add an entry mapping it to a known array_type.")
    return cfg[subject]


def resolve_pitch_for_subject(
        subject: str,
        array_pitch_path: Optional[Path] = None,
        subject_array_path: Optional[Path] = None) -> Tuple[str, float]:
    """Convenience: return ``(array_type, base_pitch_mm)`` for a subject."""
    array_type = lookup_subject_array(subject, subject_array_path)
    info = load_array_pitch(array_type, array_pitch_path)
    return array_type, float(info['pitch_mm'])


def unique_channel_count(chanmap: np.ndarray) -> int:
    """Number of distinct (non-NaN) channels in the chanmap."""
    vals = chanmap[~np.isnan(chanmap)]
    return int(np.unique(vals).size)


def all_channels(chanmap: np.ndarray) -> np.ndarray:
    """Return all unique channel indices in the chanmap, sorted."""
    vals = chanmap[~np.isnan(chanmap)]
    return np.unique(vals).astype(int)


def _macro_footprint(chanmap: np.ndarray, ch_idx: int) -> List[Tuple[int, int]]:
    """Return all cells occupied by a (possibly macro) channel index."""
    rows, cols = np.where(chanmap == ch_idx)
    return list(zip(rows.tolist(), cols.tolist()))


def macro_footprint_map(chanmap: np.ndarray) -> dict:
    """Map each channel index to the list of cells it occupies."""
    out = {}
    for ch in all_channels(chanmap):
        out[int(ch)] = _macro_footprint(chanmap, ch)
    return out
