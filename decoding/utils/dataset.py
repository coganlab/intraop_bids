from pathlib import Path

import numpy as np
from mne import read_epochs, get_montage_volume_labels
from mne.channels import make_dig_montage
from mne_bids import BIDSPath


RECON_DIR = Path("~/cworkspace/ECoG_Recon_Full/").expanduser()


def format_band_str(suffix):
    """Render a band suffix (string or sequence of strings) for filenames.

    A single-band ``suffix`` is returned unchanged.  A sequence is joined
    with ``+`` so a multi-band run is identifiable in the output path
    (e.g. ``highGamma+spikeBand``).
    """
    if isinstance(suffix, str):
        return suffix
    return "+".join(str(s) for s in suffix)


def _normalize_suffixes(suffix):
    """Return ``suffix`` as a list of strings."""
    if isinstance(suffix, str):
        return [suffix]
    return [str(s) for s in suffix]


def _collapse_hierarchical_event_id(events, event_id):
    """Collapse path-style event descriptions to their leaf token.

    Each description in ``event_id`` is reduced via
    ``name.split('/')[-1]`` (e.g. ``'response/1/AA' -> 'AA'``,
    ``'Word/Response/LS/uga' -> 'uga'``).  When several descriptions
    share a leaf -- the case introduced by saving phonemes with a
    position prefix, so that ``response/{1,2,3}/AA`` were three distinct
    classes -- their trials are merged into a single integer code and
    ``events[:, 2]`` is rewritten to match.

    Parameters
    ----------
    events : (n_events, 3) np.ndarray
        MNE events array ``(sample, prev_code, code)``.
    event_id : dict[str, int]
        Mapping from description string to integer code, as found on
        ``mne.Epochs.event_id``.

    Returns
    -------
    new_events : (n_events, 3) np.ndarray
        Copy of ``events`` with collapsed integer codes.
    new_event_id : dict[str, int]
        Mapping from leaf description to integer code.  Original codes
        are kept when the leaf mapping is already bijective (no merging
        needed); otherwise codes are reassigned in sorted-leaf order
        starting from 1 so the result is deterministic.
    """
    leaf_for = {name: name.split('/')[-1] for name in event_id.keys()}
    unique_leaves = set(leaf_for.values())

    if len(unique_leaves) == len(event_id):
        # Bijective: rename keys to their leaf token but keep integer
        # codes (and therefore events) unchanged.
        new_event_id = {leaf_for[name]: code
                        for name, code in event_id.items()}
        return events.copy(), new_event_id

    sorted_leaves = sorted(unique_leaves)
    new_event_id = {leaf: i + 1 for i, leaf in enumerate(sorted_leaves)}
    old_to_new = {event_id[name]: new_event_id[leaf_for[name]]
                  for name in event_id.keys()}
    new_events = events.copy()
    new_events[:, 2] = np.array(
        [old_to_new[int(c)] for c in events[:, 2]],
        dtype=events.dtype)
    return new_events, new_event_id


class PhonemeDatasetBIDS:
    """Load phoneme-level epoch data from a BIDS derivatives directory.

    Resolves a BIDSPath, reads MNE epochs, optionally crops to a time
    window, and slices out a specific phoneme position or returns all.

    ``suffix`` may be either a single band name (string) or a sequence of
    band names.  When multiple bands are supplied, one ``.fif`` is loaded
    per band; trials, channels, sampling rate, and event ids must match
    across bands, and the per-band features are concatenated along the
    channel axis so the returned ``X`` has shape
    ``(n_trials, n_channels * n_bands, n_times)``.

    Hierarchical/path-style event descriptions (e.g.
    ``response/<position>/<phoneme>`` for phoneme-level uECoG, or
    ``Word/Response/<lang>/<codeword>`` for word-level sEEG) are
    collapsed to their leaf token at load time.  This guarantees that
    the position-tagged phoneme labels recently produced by the BIDS
    saving pipeline are merged back into one class per phoneme, so
    ``labels`` and ``label_dict`` always describe the leaf inventory
    (e.g. 9 phonemes, not 27 position+phoneme pairs).
    """

    def __init__(self, bidsRoot, subject, phonemeIdx, nPhons=3,
                 tw=None, nan_bad_chans=False, elec_roi=None,
                 recon_dir=RECON_DIR, **kwargs):
        self.bidsRoot = Path(bidsRoot).expanduser()
        self.subject = subject
        self.phonemeIdx = 'all' if phonemeIdx == -1 else phonemeIdx
        self.nPhons = nPhons
        self.tw = tw if tw is not None else [None, None]
        self.nan_bad_chans = nan_bad_chans
        self.elec_roi = set(elec_roi) if elec_roi is not None else None
        self.recon_dir = Path(recon_dir).expanduser()
        
        suffix = kwargs.pop('suffix', None)
        self.suffixes = _normalize_suffixes(suffix) if suffix is not None \
            else [None]
        self.n_bands = len(self.suffixes)
        self._bids_kwargs = kwargs

        self.data, self.labels = self._load_data(nan_bad_chans=nan_bad_chans,
                                                 elec_roi=self.elec_roi)

    def get_data(self):
        return self.data, self.labels

    def _load_one_band(self, sfx, nan_bad_chans=False, elec_roi=None):
        bids_path = BIDSPath(
            root=self.bidsRoot,
            subject=self.subject,
            check=False,
            suffix=sfx,
            **self._bids_kwargs,
        )
        
        try:
            data_path = bids_path.match()[0]
        except IndexError:
            raise FileNotFoundError(
                f'No matching file found for {bids_path}'
            )

        epochs = read_epochs(data_path, preload=True)

        if elec_roi is not None:
            roi_picks = self._get_roi_picks(epochs, elec_roi)
            if not roi_picks:
                print(f"Warning: No electrodes found in ROIs {elec_roi} for "
                      f"subject {self.subject}. Returning empty data.")
                epochs._data = np.empty((epochs._data.shape[0], 0, epochs._data.shape[2]))
                return epochs
            epochs.pick(roi_picks)

        if nan_bad_chans:
            bad_chans = epochs.info['bads'].copy()

            bad_mask = np.isin(epochs.ch_names, bad_chans)

            # Data shape: (epochs, channels, timepoints)
            # Replace bad channels with NaNs in place. _data will exist bc preload=True
            epochs._data[:, bad_mask, :] = np.nan

        return epochs

    def _get_roi_picks(self, epochs, rois):
        elec_labels = self._get_elec_labels(epochs)
        roi_picks = self._get_roi_picks_from_labels(elec_labels, rois)
        return roi_picks

    def _get_elec_labels(self, epochs):
        montage = epochs.get_montage()
        positions = montage.get_positions()['ch_pos']
        montage_mri = make_dig_montage(ch_pos=positions, coord_frame='mri')

        # labels per electrode
        labels, _ = get_montage_volume_labels(
            montage_mri,
            subject=self.subject[0] + self.subject[1:].lstrip('0'),
            subjects_dir=self.recon_dir,
            aseg="aparc.a2009s+aseg",
            # aseg="aparc+aseg",
            dist=1
        )

        return labels

    def _get_roi_picks_from_labels(self, elec_labels, rois):
        return [
                ch_name
                for ch_name, roi_list in elec_labels.items()
                if rois.intersection(roi_list)
            ]

    def _load_data(self, nan_bad_chans=False, elec_roi=None):
        feats_per_band = []
        labels_ref = None
        ch_names_ref = None
        sfreq_ref = None
        label_dict_ref = None

        for band_idx, sfx in enumerate(self.suffixes):
            data = self._load_one_band(sfx, nan_bad_chans, elec_roi)
            t_data = data.times

            if band_idx == 0:
                self.twEpoch = [
                    t_data[0] if self.tw[0] is None else self.tw[0],
                    t_data[-1] if self.tw[1] is None else self.tw[1],
                ]
            data = data.crop(tmin=self.twEpoch[0], tmax=self.twEpoch[1])

            # Merge hierarchical event descriptions
            # (e.g. ``response/1/AA``, ``response/2/AA``, ``response/3/AA``)
            # into a single class per leaf token *before* slicing, so
            # trial codes downstream describe just the phoneme.  The
            # interleaved trial order ``[w1_p1, w1_p2, ..., w2_p1, ...]``
            # is preserved by ``mne.read_epochs``, so the position-based
            # slicing below remains correct.
            events, event_id = _collapse_hierarchical_event_id(
                data.events, data.event_id)

            if self.phonemeIdx == 'all':
                features = data.get_data()
                labels = events[:, 2]
            elif self.phonemeIdx < 1 or self.phonemeIdx > self.nPhons:
                raise ValueError(
                    f'phonemeIdx must be between 1 and {self.nPhons}, '
                    f'got {self.phonemeIdx}')
            else:
                features = data.get_data()[(self.phonemeIdx - 1)::self.nPhons]
                labels = events[(self.phonemeIdx - 1)::self.nPhons, 2]

            if band_idx == 0:
                labels_ref = labels
                ch_names_ref = list(data.ch_names)
                sfreq_ref = float(data.info['sfreq'])
                label_dict_ref = {v: k for k, v in event_id.items()}
            else:
                if not np.array_equal(labels, labels_ref):
                    raise ValueError(
                        f"Label mismatch between band '{self.suffixes[0]}' "
                        f"and '{sfx}'. All bands must share identical trial "
                        f"ordering and event ids.")
                if list(data.ch_names) != ch_names_ref:
                    raise ValueError(
                        f"Channel name mismatch between band "
                        f"'{self.suffixes[0]}' and '{sfx}'.")
                sfreq_i = float(data.info['sfreq'])
                if sfreq_i != sfreq_ref:
                    raise ValueError(
                        f"Sampling rate mismatch between band "
                        f"'{self.suffixes[0]}' ({sfreq_ref} Hz) and "
                        f"'{sfx}' ({sfreq_i} Hz).")

            feats_per_band.append(features)

        features = (feats_per_band[0] if self.n_bands == 1
                    else np.concatenate(feats_per_band, axis=1))

        self.ch_names_per_band = ch_names_ref
        self.n_ch_per_band = len(ch_names_ref)
        self.ch_names = ch_names_ref
        self.sfreq = sfreq_ref
        self.label_dict = label_dict_ref
        return features, labels_ref
