from pathlib import Path
from mne_bids import BIDSPath
from mne import read_epochs


class PhonemeDatasetBIDS:
    """Load phoneme-level epoch data from a BIDS derivatives directory.

    Resolves a BIDSPath, reads MNE epochs, optionally crops to a time
    window, and slices out a specific phoneme position or returns all.
    """

    def __init__(self, bidsRoot, subject, phonemeIdx, nPhons=3,
                 tw=None, **kwargs):
        self.bidsRoot = Path(bidsRoot)
        self.subject = subject
        self.phonemeIdx = 'all' if phonemeIdx == -1 else phonemeIdx
        self.nPhons = nPhons
        self.tw = tw if tw is not None else [None, None]

        self.bidsPath = BIDSPath(
            root=self.bidsRoot,
            subject=self.subject,
            check=False,
            **kwargs
        )
        self.data, self.labels = self._load_data()

    def get_data(self):
        return self.data, self.labels

    def _load_data(self):
        try:
            dataPath = self.bidsPath.match()[0]
        except IndexError:
            raise FileNotFoundError(
                f'No matching file found for {self.bidsPath}')

        data = read_epochs(dataPath, preload=True)
        t_data = data.times

        self.twEpoch = [t_data[0] if self.tw[0] is None else self.tw[0],
                        t_data[-1] if self.tw[1] is None else self.tw[1]]
        data = data.crop(tmin=self.twEpoch[0], tmax=self.twEpoch[1])

        if self.phonemeIdx == 'all':
            features = data.get_data()
            labels = data.events[:, 2]
        elif self.phonemeIdx < 1 or self.phonemeIdx > self.nPhons:
            raise ValueError(f'phonemeIdx must be between 1 and {self.nPhons},'
                             f' got {self.phonemeIdx}')
        else:
            features = data.get_data()[(self.phonemeIdx - 1)::self.nPhons]
            labels = data.events[(self.phonemeIdx - 1)::self.nPhons, 2]

        self.label_dict = {v: k for k, v in data.event_id.items()}
        return features, labels
