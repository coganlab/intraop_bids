from pathlib import Path

import numpy as np
from mne import read_epochs
from mne_bids import BIDSPath


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
    """

    def __init__(self, bidsRoot, subject, phonemeIdx, nPhons=3,
                 tw=None, **kwargs):
        self.bidsRoot = Path(bidsRoot)
        self.subject = subject
        self.phonemeIdx = 'all' if phonemeIdx == -1 else phonemeIdx
        self.nPhons = nPhons
        self.tw = tw if tw is not None else [None, None]

        suffix = kwargs.pop('suffix', None)
        self.suffixes = _normalize_suffixes(suffix) if suffix is not None \
            else [None]
        self.n_bands = len(self.suffixes)
        self._bids_kwargs = kwargs

        self.data, self.labels = self._load_data()

    def get_data(self):
        return self.data, self.labels

    def _load_one_band(self, sfx):
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
                f'No matching file found for {bids_path}')
        return read_epochs(data_path, preload=True)

    def _load_data(self):
        feats_per_band = []
        labels_ref = None
        ch_names_ref = None
        sfreq_ref = None
        label_dict_ref = None

        for band_idx, sfx in enumerate(self.suffixes):
            data = self._load_one_band(sfx)
            t_data = data.times

            if band_idx == 0:
                self.twEpoch = [
                    t_data[0] if self.tw[0] is None else self.tw[0],
                    t_data[-1] if self.tw[1] is None else self.tw[1],
                ]
            data = data.crop(tmin=self.twEpoch[0], tmax=self.twEpoch[1])

            if self.phonemeIdx == 'all':
                features = data.get_data()
                labels = data.events[:, 2]
            elif self.phonemeIdx < 1 or self.phonemeIdx > self.nPhons:
                raise ValueError(
                    f'phonemeIdx must be between 1 and {self.nPhons}, '
                    f'got {self.phonemeIdx}')
            else:
                features = data.get_data()[(self.phonemeIdx - 1)::self.nPhons]
                labels = data.events[(self.phonemeIdx - 1)::self.nPhons, 2]

            if band_idx == 0:
                labels_ref = labels
                ch_names_ref = list(data.ch_names)
                sfreq_ref = float(data.info['sfreq'])
                label_dict_ref = {v: k for k, v in data.event_id.items()}
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
