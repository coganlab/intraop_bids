from pathlib import Path
from mne_bids import BIDSPath
from mne import read_epochs
from torch.utils.data import Dataset

class PhonemeDatasetBIDS(Dataset):

    def __init__(self, bidsRoot, subject, phonemeIdx, nPhons=3,
                 tw=[None, None], **kwargs):
        self.bidsRoot = Path(bidsRoot)
        self.subject = subject
        self.phonemeIdx = 'all' if phonemeIdx == -1 else phonemeIdx
        self.nPhons = nPhons
        self.tw = tw

        self.bidsPath = BIDSPath(
            root=self.bidsRoot,
            subject=self.subject,
            check=False,
            **kwargs
        )
        self.data, self.labels = self._load_data()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_data(self):
        return self.data, self.labels
    
    def _load_data(self):
        try:
            dataPath = self.bidsPath.match()[0]
        except IndexError:
            raise FileNotFoundError('No matching file found for'
                                    f'{self.bidsPath}')
        # extract data and labels from epochs object
        data = read_epochs(dataPath, preload=True)
        t_data = data.times

        # trim time window (takes default bound if None)
        self.twEpoch = [t_data[0] if self.tw[0] is None else self.tw[0],
                         t_data[-1] if self.tw[1] is None else self.tw[1]]
        data = data.crop(tmin=self.twEpoch[0], tmax=self.twEpoch[1])

        # no need to cut out specific phonemes if using all
        if self.phonemeIdx == 'all':
            features = data.get_data()
            labels = data.events[:, 2]
        # check phonemeIdx is valid
        elif self.phonemeIdx < 1 or self.phonemeIdx > self.nPhons:
            raise ValueError(f'phonemeIdx must be between 1 and {self.nPhons},'
                             f'got {self.phonemeIdx}')
        # otherwise, slice out the relevant phoneme
        else:
            features = data.get_data()[(self.phonemeIdx-1)::self.nPhons]
            # shape (n_epochs,)
            labels = data.events[(self.phonemeIdx-1)::self.nPhons, 2]

        # Create a mapping from phoneme index to label (reverse mapping of
        # event_id)
        self.label_dict = swap_kv_dict(data.event_id)
        return features, labels

def swap_kv_dict(d):
    return dict((v, k) for k, v in d.items())