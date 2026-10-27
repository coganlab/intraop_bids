"""
ECoG/iEEG to BIDS conversion utilities built on mne-bids.

Pipeline overview
- Load raw Intan RHD data via ``dataloaders.rhdLoader``, which handles
  loading, downsampling, trigger detection, and MFA alignment.
- Build an MNE Raw object from the processed HDF5 output and attach
  word-level annotations from MFA token files.
- Ensure the raw has a montage (RAS-based or synthetic placeholder).
- Write iEEG data to a BIDS directory tree with ``write_raw_bids()``.
- Post-process the auto-generated events.tsv to a standardised format
  with columns: subject, trial, onset, duration, value, trial_type, sample.
- Write phoneme-level events and a copy of the raw data to a
  ``derivatives/phonemeLevel/`` folder for phoneme-level analysis.
- Convert anatomical data (T1w and CT when present) into ``anat/``.
- Optionally save microphone audio as a derivative.

Notes and assumptions
- The RAS montage file is expected to be whitespace-delimited with columns:
  prefix, number, x, y, z, hemisphere, grid.
"""
import json
import logging
import re
from pathlib import Path
from typing import Optional, Union

import hydra
import h5py
import mne
import numpy as np
import scipy.io as sio
import os
import pandas as pd
from montreal_forced_aligner.models import DictionaryModel
from mne_bids import BIDSPath, write_raw_bids, write_anat
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

TASK2MFA = {
    "lexical": "lexical_repeat_intraop",
    "phoneme": "phoneme_sequencing",
    "sentence": "sentence_repeat_intraop",
}

TASK2MFA_DICT = {
    "lexical": "english_us_lr",
    "phoneme": "english_us_ps",
    "sentence": "english_us_arpa",
}

TASK_RESP_WINS = {
    "lexical": 3.0,
    "phoneme": 3.0,
    "sentence": 5.5,
}

# NPHONS = {
#     "lexical": 5,
#     "phoneme": 3,
# }

PS2ARPA = {
    'a': 'AA', 'ae': 'EH', 'i': 'IY', 'u': 'UW', 'b': 'B', 'p': 'P',
    'v': 'V', 'g': 'G', 'k': 'K', 'UH': 'UW', 'AE': 'EH',
}

EVENTS_COL_ORDER = [
    'subject', 'trial', 'onset', 'duration', 'value', 'trial_type', 'sample',
]

EVENTS_JSON_METADATA = {
    "trial": {"Description": "Trial number within the task"},
    "onset": {"Description": "Onset time of the event in seconds", "Units": "s"},
    "duration": {
        "Description": "Duration of the event in seconds", "Units": "s",
    },
    "value": {"Description": "Word label for the event"},
    "trial_type": {
        "Description": "Type of event",
        "Levels": {
            "stimulus": "Auditory stimulus presentation",
            "response": "Subject verbal response",
        },
    },
    "sample": {"Description": "Onset sample index in the raw data"},
}

PHONEME_EVENTS_JSON_METADATA = {
    "trial": {"Description": "Trial number within the task"},
    "onset": {
        "Description": "Onset time of the phoneme event in seconds",
        "Units": "s",
    },
    "duration": {
        "Description": "Duration of the phoneme event in seconds",
        "Units": "s",
    },
    "value": {"Description": "Phoneme label (ARPAbet) for the event"},
    "trial_type": {
        "Description": (
            "Type and ordinal position of the phoneme within its trial "
            "(e.g. stimulus/1 for the first phoneme of the stimulus)"
        ),
    },
    "sample": {"Description": "Onset sample index in the raw data"},
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def remove_arpabet_stress(phoneme: str) -> str:
    """Strip numeric stress markers from an ARPAbet phoneme (e.g. AH0 -> AH)."""
    return re.sub(r'\d', '', phoneme)


def remove_orphan_stimuli(events_df, response_window_sec=3.0):
    """Drop stimulus rows that lack a matching response within *response_window_sec*."""
    base_types = events_df['trial_type'].str.split('/').str[0]
    is_stim = base_types == 'stimulus'
    is_resp = base_types == 'response'

    stim_onsets = events_df.loc[is_stim, 'onset'].values
    resp_onsets = events_df.loc[is_resp, 'onset'].values

    if len(resp_onsets) == 0:
        return events_df[~is_stim].reset_index(drop=True)

    has_response = np.any(
        (resp_onsets[None, :] > stim_onsets[:, None])
        & (resp_onsets[None, :] <= stim_onsets[:, None] + response_window_sec),
        axis=1,
    )

    orphan_mask = pd.Series(False, index=events_df.index)
    orphan_mask.iloc[np.where(is_stim)[0][~has_response]] = True
    return events_df[~orphan_mask].reset_index(drop=True)


def subevent_txt2df(txt_path, event_type, trial_type, subject, n_items_per_trial, fs, task):
    """Convert a sub-event MFA text file to a BIDS events DataFrame."""
    df = pd.read_csv(txt_path, sep='\t', names=['onset', 'offset', 'value'])
    df['duration'] = df['offset'] - df['onset']

    if task == 'phoneme' and event_type == 'phones':
        df['value'] = df['value'].apply(lambda x: PS2ARPA.get(x, x))
    if event_type == 'phones':
        df['value'] = df['value'].apply(remove_arpabet_stress)

    df['subject'] = subject
    df['sample'] = (df['onset'] * fs).astype(int)
    # df['trial'] = (np.arange(len(df)) // n_phons) + 1
    # position = (np.arange(len(df)) % n_phons) + 1
    # df['trial_type'] = [f'{trial_type}/{p}' for p in position]

    trial_nums, position_num = items2trials(len(df), n_items_per_trial)
    df['trial'] = trial_nums
    df['trial_type'] = [f'{trial_type}/{p}' for p in position_num]

    return df[EVENTS_COL_ORDER]


def items2trials(n_items_total, n_items_per_trial):
    """Assign trial numbers and positions to sequential annotations."""
    n_items_per_trial = np.asarray(n_items_per_trial, dtype=int)

    if np.any(n_items_per_trial < 0):
        raise ValueError("Item counts cannot be negative.")

    expected = n_items_per_trial.sum()

    if n_items_total != expected:
        raise ValueError(
            f"Found {n_items_total} annotations, but trial counts "
            f"sum to {expected}."
        )

    trial_nums = np.repeat(
        np.arange(1, len(n_items_per_trial) + 1),
        n_items_per_trial,
    )

    position_num = np.concatenate([
        np.arange(1, n + 1)
        for n in n_items_per_trial
    ]) if expected else np.array([], dtype=int)

    return trial_nums.tolist(), position_num.tolist()


def _parse_missing_data_idx(missing_data_idx, type = 'list'):
    """Parse the missing data index from the configuration."""
    if missing_data_idx is None:
        return None
    if type == 'list' and isinstance(missing_data_idx, list):
        return missing_data_idx
    elif type == 'range':
        if isinstance(missing_data_idx, range):
            return list(missing_data_idx)
        else:
            return list(range(missing_data_idx[0], missing_data_idx[-1] + 1))
    else:
        raise ValueError(f"Invalid missing data index type: {type}")

try:
    from subject_utils import base_subject_id
except ImportError:
    from .subject_utils import base_subject_id


def get_words_per_trial(trial_info_path: Union[str, Path]) -> list[int]:
    """Get the number of words in each trial from the trial info file."""
    if not os.path.exists(trial_info_path):
        raise FileNotFoundError(f"Trial info file not found: {trial_info_path}")
    trial_info = sio.loadmat(trial_info_path)['trialInfo'][0,:]
    
    words_per_trial = []
    for tr in range(len(trial_info)):
        # format changes depending on task
        try:
            curr_words = trial_info[tr]['sound'][0,0]['name'][0,0][0]
        except IndexError:
            curr_words = trial_info[tr]['sound'][0,0][0]

        # separates hyphenated words to list of words and removes extension
        curr_words = curr_words.split('_')[-1].split('.')[0].replace('-', ' ').split()

        # put single words in list to match sentence format
        if isinstance(curr_words, str):
            curr_words = [curr_words]

        words_per_trial.append(curr_words)

    return words_per_trial


import pandas as pd
import numpy as np


def get_phons_per_trial(
    cue_events_path,
    phones_path,
    trial_type,
    onset_buffer=0.02,
    offset_buffer=0.02,
):
    """Count aligned phonemes per trial using cue event timestamps."""
    cues = pd.read_csv(
        cue_events_path,
        sep='\t',
        header=None,
        names=['onset', 'offset', 'trial_id'],
    )

    phones = pd.read_csv(
        phones_path,
        sep='\t',
        header=None,
        names=['onset', 'offset', 'phoneme'],
    )

    phones = phones[
        phones['phoneme'].notna()
        & phones['phoneme'].astype(str).str.strip().ne('')
    ].copy()

    counts = []

    for i, cue in cues.iterrows():
        if trial_type == 'stimulus':
            start = cue['onset'] - onset_buffer
            end = cue['offset'] + offset_buffer

        elif trial_type == 'response':
            start = cue['offset'] - onset_buffer

            if i + 1 < len(cues):
                end = cues.iloc[i + 1]['onset'] + offset_buffer
            else:
                end = phones['offset'].max() + 1e-9

        else:
            raise ValueError(
                "trial_type must be 'stimulus' or 'response'"
            )

        trial_phones = phones[
            (phones['onset'] >= start)
            & (phones['onset'] < end)
        ]

        counts.append(len(trial_phones))

    return counts

# ---------------------------------------------------------------------------
# BIDSConverter
# ---------------------------------------------------------------------------

class BIDSConverter:
    """Converter for writing iEEG/ECoG data to BIDS.

    Responsibilities
    - Manage source paths, subject/task identifiers, and optional audio/anat paths.
    - Process events and ensure ``raw`` has a montage (RAS-based or synthetic).
    - Create a BIDS directory structure and write iEEG data using ``mne-bids``.
    - Post-process events.tsv to a standardised column format.
    - Write phoneme-level events and raw data to a derivatives folder.
    - Convert available anatomical images (T1w, CT) and create placeholder
      JSON sidecars.
    """

    def __init__(
        self,
        subject: str,
        task: str,
        source_path: Union[str, Path],
        bids_root: Optional[Union[str, Path]] = None,
        recon_path: Optional[Union[str, Path]] = None,
    ):
        self.subject = subject
        self.recon_subject = base_subject_id(subject)
        self.task = task
        self.source_path = Path(source_path)
        self.recon_path = Path(recon_path)
        self.bids_root = Path(bids_root)

        self.experiment_info = None
        self.raw = None
        self.events = None
        self.bids_path = None

    # ------------------------------------------------------------------
    # Raw object construction
    # ------------------------------------------------------------------

    def _build_channel_index_to_name(self):
        """Invert the channel map to map each channel index to an ``R{row}-C{col}`` name.

        For hybrid arrays where a macro electrode occupies multiple grid
        positions, the **first** position encountered in row-major order
        (top-left corner) is kept as the canonical name.  A companion
        dict ``self._ch_index_to_type`` is built at the same time,
        mapping each channel index to ``"micro"`` or ``"macro"``.
        """
        index_to_name = {}
        index_count: dict[int, int] = {}
        for row_idx in range(self.channel_map.shape[0]):
            for col_idx in range(self.channel_map.shape[1]):
                ch_num = self.channel_map[row_idx, col_idx]
                if not np.isnan(ch_num):
                    key = int(ch_num)
                    index_count[key] = index_count.get(key, 0) + 1
                    if key not in index_to_name:
                        index_to_name[key] = f"R{row_idx}-C{col_idx}"
        self._ch_index_to_type = {
            k: "macro" if index_count[k] > 1 else "micro"
            for k in index_to_name
        }
        return index_to_name

    def _make_raw_object(self):
        """Build an MNE RawArray from the preprocessed HDF5 file."""
        subj_dir = os.path.join(
            self.source_path, f'sub-{self.subject}',
        )
        raw_path = os.path.join(subj_dir, f'sub-{self.subject}_raw.h5')
        raw = h5py.File(raw_path, 'r')

        raw_data = raw['data'][()]
        self.fs = raw['fs'][()]

        standalone_map = os.path.join(
            subj_dir, f'sub-{self.subject}_channel_map.npy',
        )
        if os.path.exists(standalone_map):
            self.channel_map = np.load(standalone_map)
        else:
            self.channel_map = raw['channel_map'][()]

        self._ch_index_to_name = self._build_channel_index_to_name()
        channel_names = [
            self._ch_index_to_name.get(i, f'{i}')
            for i in range(len(raw_data))
        ]
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.fs,
            ch_types='ecog',
        )
        bad_channel_indices = {int(ch) for ch in raw['bad_channels'][()]}
        if 'missing_channels' in raw:
            bad_channel_indices.update(int(ch) for ch in raw['missing_channels'][()])
        non_finite_channels = np.where(~np.all(np.isfinite(raw_data), axis=1))[0]
        bad_channel_indices.update(int(ch) for ch in non_finite_channels)
        info['bads'] = [
            self._ch_index_to_name.get(ch, str(ch))
            for ch in sorted(bad_channel_indices)
        ]
        if non_finite_channels.size:
            logger.info(
                'Replacing non-finite placeholder samples with 0 in %d '
                'channel(s) marked bad for BIDS export.',
                non_finite_channels.size,
            )
        raw.close()

        raw_data = np.where(np.isfinite(raw_data), raw_data, 0.0).astype(np.float32)
        raw = mne.io.RawArray(data=raw_data * 1e-6, info=info)
        del raw_data
        import gc
        gc.collect()

        return raw

    # ------------------------------------------------------------------
    # Post-write electrode type patching
    # ------------------------------------------------------------------

    def _add_electrode_type_to_tsvs(self, bids_path, verbose=True):
        """Append an ``electrode_type`` column to channels and electrodes TSVs.

        Must be called after ``write_raw_bids`` so the TSV files exist on
        disk.  Uses ``self._ch_index_to_name`` to map row positions back
        to channel indices and ``self._ch_index_to_type`` for the type.
        """
        name_to_type = {
            name: self._ch_index_to_type.get(idx, "micro")
            for idx, name in self._ch_index_to_name.items()
        }
        subject_id = bids_path.subject
        for suffix in ("channels", "electrodes"):
            tsv_files = list(
                Path(bids_path.root).rglob(
                    f"sub-{subject_id}*_{suffix}.tsv",
                ),
            )
            for tsv_path in tsv_files:
                df = pd.read_csv(tsv_path, sep="\t")
                if "name" not in df.columns:
                    continue
                df["electrode_type"] = df["name"].map(name_to_type).fillna(
                    "micro",
                )
                df.to_csv(tsv_path, sep="\t", index=False)
                if verbose:
                    print(
                        f"Added electrode_type column to: {tsv_path}"
                    )

    # ------------------------------------------------------------------
    # Annotations / montage
    # ------------------------------------------------------------------

    def update_raw_object(self, ann):
        """Set annotations on *self.raw* and ensure a montage is present."""
        self.raw.set_annotations(ann)
        if self.raw.info.get_montage() is None:
            self.raw = self.add_montage_to_raw()

    def add_montage_to_raw(self):
        """Add a montage using the subject RAS file, falling back to a synthetic grid."""
        ras_file = os.path.join(
            self.recon_path,
            self.recon_subject,
            "elec_recon",
            f"{self.recon_subject}_elec_locations_RAS_brainshifted.txt",
        )

        if not os.path.exists(ras_file):
            print(
                f"RAS file not found: {ras_file}, "
                "using custom montage as placeholder"
            )
            montage = self.make_custom_montage()
            self.raw.set_montage(montage)
            return self.raw
        try:
            montage = self.load_ras_montage(ras_file)
            self.raw.set_montage(montage)
        except Exception as e:
            print(f"Error applying RAS montage: {e}")
            print("Falling back to custom montage...")
            montage = self.make_custom_montage()
            self.raw.set_montage(montage)

        return self.raw

    def load_ras_montage(self, ras_file):
        """Load an electrode RAS montage from a whitespace-delimited text file."""
        from mne.channels import make_dig_montage
        from mne.io.constants import FIFF

        df = pd.read_csv(
            ras_file,
            sep=r'\s+',
            header=None,
            names=['prefix', 'number', 'x', 'y', 'z', 'hemisphere', 'grid'],
        )

        orig_nums = df['number'].astype(int).tolist()
        unique_sorted = sorted(set(orig_nums))
        num_map = {old: i for i, old in enumerate(unique_sorted)}
        ch_names = [
            self._ch_index_to_name.get(num_map[n], str(num_map[n]))
            for n in orig_nums
        ]
        pos = df[['x', 'y', 'z']].values

        montage = make_dig_montage(
            ch_pos=dict(zip(ch_names, pos)),
            coord_frame='ras',
        )

        for ch, hemi in zip(montage.dig, df['hemisphere']):
            if ch['kind'] == FIFF.FIFFV_POINT_EEG:
                ch['hemisphere'] = hemi.upper()

        return montage

    def make_custom_montage(self):
        """Create a synthetic 2D-grid montage as a placeholder."""
        from mne.channels import Layout, make_dig_montage

        channel_map = self.channel_map
        n_rows, n_cols = channel_map.shape
        pos, names = [], []

        for i in range(n_rows):
            for j in range(n_cols):
                if not np.isnan(channel_map[i, j]):
                    x = (j + 0.5) / n_cols
                    y = (n_rows - i - 0.5) / n_rows
                    width = 1 / n_cols
                    height = 1 / n_rows
                    pos.append([x, y, width, height])
                    names.append(f'R{i}-C{j}')

        pos = np.array(pos)
        box = np.array([0, 0, 1, 1])
        layout = Layout(
            pos=pos, names=names, kind='custom',
            ids=np.arange(1, len(names) + 1), box=box,
        )

        ch_pos = {
            name: [x, y, 0]
            for name, (x, y, _, _) in zip(layout.names, layout.pos)
        }
        montage = make_dig_montage(ch_pos, coord_frame='ras')
        return montage

    # ------------------------------------------------------------------
    # Event parsing
    # ------------------------------------------------------------------

    def _parse_events(self, sfreq):
        """Build MNE Annotations from word-level MFA text files."""
        subj_dir = os.path.join(self.source_path, f'sub-{self.subject}')
        # will try to load corrected files first, then fallback to original files
        stim_file = [os.path.join(subj_dir, "mfa_adj_stim_words.txt"), os.path.join(subj_dir, "mfa_stim_words.txt")]
        resp_file = [os.path.join(subj_dir, "mfa_adj_resp_words.txt"), os.path.join(subj_dir, "mfa_resp_words.txt")]

        onsets_sec, durations_sec, descriptions = [], [], []

        stim_words = None
        if os.path.exists(stim_file[0]):
            stim_words = np.loadtxt(stim_file[0], dtype=str)
            logger.info(f"Loaded corrected stimulus word file from {stim_file[0]}")
        elif os.path.exists(stim_file[1]):
            stim_words = np.loadtxt(stim_file[1], dtype=str)
            logger.info(f"Loaded original stimulus word file from {stim_file[1]}")
        else:
            logger.error(f"Stimulus word file not found in {subj_dir}")
            raise FileNotFoundError(f"Stimulus word file not found in {subj_dir}")

        if stim_words is not None:
            stim_contents = self._build_trial_events_from_words(stim_words)
            for row in stim_contents:
                    onset_sec, offset_sec, word = row[0], row[1], row[2]
                    onsets_sec.append(float(onset_sec))
                    durations_sec.append(
                        max(0.0, float(offset_sec) - float(onset_sec))
                    )
                    descriptions.append(f"stimulus/{word}")

        resp_words = None
        if os.path.exists(resp_file[0]):
            resp_words = np.loadtxt(resp_file[0], dtype=str)
            logger.info(f"Loaded corrected response word file from {resp_file[0]}")
        elif os.path.exists(resp_file[1]):
            resp_words = np.loadtxt(resp_file[1], dtype=str)
            logger.info(f"Loaded original response word file from {resp_file[1]}")
        else:
            logger.error(f"Response word file not found in {subj_dir}")
            raise FileNotFoundError(f"Response word file not found in {subj_dir}")

        if resp_words is not None:
            resp_contents = self._build_trial_events_from_words(resp_words)
            for row in resp_contents:
                onset_sec, offset_sec, word = row[0], row[1], row[2]
                onsets_sec.append(float(onset_sec))
                durations_sec.append(
                    max(0.0, float(offset_sec) - float(onset_sec))
                )
                descriptions.append(f"response/{word}")

        return mne.Annotations(
            onset=onsets_sec,
            duration=durations_sec,
            description=descriptions,
            orig_time=self.raw.info.get('meas_date'),
        )

    def _build_trial_events_from_words(self, word_timings):
        """Convert word annotations to trial-level events"""
        # get info on words in a single trial
        trial_info_file = self.source_path / f'sub-{self.subject}' / 'trialInfo.mat'
        words_per_trial = get_words_per_trial(trial_info_file)
        n_words_per_trial = [len(words) for words in words_per_trial]

        word_timings_pop = word_timings.copy()
        trial_events = []
        for i, n in enumerate(n_words_per_trial):
            trial_rows = word_timings_pop[:n]
            # check that expected words are present in the trial
            trial_words = [row[2] for row in trial_rows]
            if trial_words != words_per_trial[i]:
                logger.warning(
                    f"Trial {i+1} words do not match expected words. "
                    f"Expected: {words_per_trial[i]}, Found: {trial_words}"
                )
                continue

            # remove the words for this trial from the list for the next iteration
            word_timings_pop = word_timings_pop[n:]

            # save trial level information
            onset = trial_rows[0][0]
            offset = trial_rows[-1][1]
            trial_contents = '-'.join(trial_words)
            trial_events.append([onset, offset, trial_contents])

        return trial_events

    # ------------------------------------------------------------------
    # Post-processing: standardise events.tsv
    # ------------------------------------------------------------------

    def _reformat_events_tsv(self):
        """Rewrite the MNE-generated events.tsv to the standardised format.

        Transforms the auto-generated columns produced by ``write_raw_bids()``
        (onset, duration, trial_type="stimulus/WORD", value, sample) into the
        canonical layout used across the pipeline
        (subject, trial, onset, duration, value, trial_type, sample).
        """
        events_tsv_path = self.bids_path.copy().update(
            suffix='events', extension='.tsv',
        )
        events_df = pd.read_csv(events_tsv_path.fpath, sep='\t')

        split = events_df['trial_type'].str.split('/', n=1, expand=True)
        events_df['trial_type'] = split[0]
        events_df['value'] = split[1]
        events_df['subject'] = self.subject

        events_df = events_df.sort_values('onset').reset_index(drop=True)
        events_df = remove_orphan_stimuli(events_df, response_window_sec=TASK_RESP_WINS.get(self.task, 3.0))

        events_df['trial'] = np.ceil(
            np.arange(1, len(events_df) + 1) / 2
        ).astype(int)

        events_df = events_df[EVENTS_COL_ORDER]
        events_df.to_csv(events_tsv_path.fpath, sep='\t', index=False)

        self._write_events_json(
            events_tsv_path.fpath.with_suffix('.json'),
            EVENTS_JSON_METADATA,
        )

    # ------------------------------------------------------------------
    # Sub-event derivative
    # ------------------------------------------------------------------

    def _write_subevent_derivative(self, event_type='phones', overwrite=True, verbose=True):
        """Copy raw data and write phoneme-level events to derivatives/phonemeLevel/.
    
        The derivative folder is treated as its own BIDSLayout by downstream
        preprocessing code, so the raw data file must be present alongside
        the events.
        """
        from bids import BIDSLayout
        from ieeg.io import save_derivative

        if event_type == 'phones':
            deriv_name = 'phonemeLevel'
        elif event_type == 'words':
            deriv_name = 'wordLevel'
        else:
            raise ValueError(f"Invalid event_type: {event_type}. Must be 'phones' or 'words'.")

        bids_layout = BIDSLayout(
            root=str(self.bids_root), derivatives=True,
        )
        save_derivative(self.raw, bids_layout, deriv_name, overwrite)

        if hasattr(self, '_ch_index_to_type'):
            deriv_path = BIDSPath(
                root=self.bids_root / 'derivatives' / 'deriv_name',
                subject=self.subject,
                task=self.task,
                datatype='ieeg',
                suffix='ieeg',
                check=False,
            )
            self._add_electrode_type_to_tsvs(deriv_path, verbose=verbose)

        subj_dir = self.source_path / f'sub-{self.subject}'
        stim_item_file = [subj_dir / f'mfa_adj_stim_{event_type}.txt', subj_dir / f'mfa_stim_{event_type}.txt']
        resp_item_file = [subj_dir / f'mfa_adj_resp_{event_type}.txt', subj_dir / f'mfa_resp_{event_type}.txt']

        if stim_item_file[0].exists():
            stim_fname = stim_item_file[0]
        elif stim_item_file[1].exists():
            stim_fname = stim_item_file[1]
        else:
            logger.error(f"Stimulus {event_type} file not found in {subj_dir}")
            raise FileNotFoundError(f"Stimulus {event_type} file not found in {subj_dir}")

        if resp_item_file[0].exists():
            resp_fname = resp_item_file[0]
        elif resp_item_file[1].exists():
            resp_fname = resp_item_file[1]
        else:
            logger.error(f"Response {event_type} file not found in {subj_dir}")
            raise FileNotFoundError(f"Response {event_type} file not found in {subj_dir}")

        trial_info_file = subj_dir / 'trialInfo.mat'
        words_per_trial = get_words_per_trial(trial_info_file)
        n_words_per_trial = [len(words) for words in words_per_trial]

        sfreq = self.raw.info['sfreq']

        if event_type == 'phones':
            cue_events_path = subj_dir / 'cue_events.txt'
            n_items_per_trial_stim = get_phons_per_trial(
                cue_events_path,
                stim_fname,
                'stimulus',
            )
            n_items_per_trial_resp = get_phons_per_trial(
                cue_events_path,
                resp_fname,
                'response',
            )
        else:
            n_items_per_trial_stim = n_words_per_trial
            n_items_per_trial_resp = n_words_per_trial

        n_items_per_trial_stim = np.array(n_items_per_trial_stim)
        n_items_per_trial_resp = np.array(n_items_per_trial_resp)

        stim_df = subevent_txt2df(
            stim_fname, event_type, 'stimulus', self.subject,
            n_items_per_trial_stim, sfreq, self.task,
        )
        stim_df['trial'] = items2trials(len(stim_df), n_items_per_trial_stim)[0]
        resp_df = subevent_txt2df(
            resp_fname, event_type, 'response', self.subject,
            n_items_per_trial_resp, sfreq, self.task,
        )
        resp_df['trial'] = items2trials(len(resp_df), n_items_per_trial_resp)[0]

        events_df = (
            pd.concat([stim_df, resp_df])
            .sort_values('onset')
            .reset_index(drop=True)
        )
        events_df = remove_orphan_stimuli(events_df, response_window_sec=TASK_RESP_WINS.get(self.task, 3.0))

        events_path = BIDSPath(
            root=self.bids_root / 'derivatives' / deriv_name,
            subject=self.subject,
            task=self.task,
            datatype='ieeg',
            suffix='events',
            description=deriv_name,
            extension='.tsv',
            check=False,
        )
        events_path.mkdir(exist_ok=True)
        events_df.to_csv(events_path.fpath, sep='\t', index=False)

        self._write_events_json(
            events_path.fpath.with_suffix('.json'),
            PHONEME_EVENTS_JSON_METADATA,
        )

        if verbose:
            logger.info(
                "Wrote %s events to %s", deriv_name, events_path.fpath,
            )

    # ------------------------------------------------------------------
    # JSON sidecar helper
    # ------------------------------------------------------------------

    @staticmethod
    def _write_events_json(json_path, metadata):
        """Write an events.json sidecar describing the events.tsv columns."""
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Derivative writing (audio, channel map, etc.)
    # ------------------------------------------------------------------

    def save_to_derivative(
        self,
        data,
        folder: str,
        filename: str,
        file_type: str = 'wav',
        description: str = 'raw',
        overwrite: bool = True,
        **kwargs,
    ) -> Optional[Path]:
        """Save arrays (e.g. audio) into ``derivatives/<folder>/``."""
        derivative_dir = self.bids_path.copy()
        derivative_dir.update(
            root=self.bids_root / "derivatives" / folder,
            suffix=filename,
            datatype=filename,
            description=description,
            extension=f'.{file_type}',
            check=False,
        )

        verbose = bool(kwargs.get('verbose', False))

        if derivative_dir.fpath.exists() and not overwrite:
            if verbose:
                print(
                    f"Derivative already exists, skipping: "
                    f"{derivative_dir.fpath}"
                )
            return derivative_dir

        derivative_dir.mkdir(exist_ok=True)

        if file_type == 'wav':
            import soundfile as sf
            sf.write(str(derivative_dir.fpath), data, self.audio_fs)
        elif file_type == 'tsv':
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            data.to_csv(derivative_dir.fpath, sep='\t', index=False)
        else:
            raise ValueError(f"Unsupported derivative file type: {file_type}")

        return derivative_dir

    # ------------------------------------------------------------------
    # Main conversion entry point
    # ------------------------------------------------------------------

    def convert_to_bids(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> BIDSPath:
        """Convert loaded data to BIDS, then write word + phoneme events."""
        bids_root = Path(output_dir) if output_dir else self.bids_root
        if bids_root is None:
            raise ValueError(
                "output_dir must be provided or bids_root must be set"
            )

        bids_path = BIDSPath(
            subject=self.subject.lstrip('sub-'),
            task=self.task,
            root=str(bids_root),
            datatype='ieeg',
            suffix='ieeg',
        )
        bids_path.mkdir(exist_ok=True)

        self.raw = self._make_raw_object()
        ann = self._parse_events(self.raw.info['sfreq'])
        self.update_raw_object(ann=ann)

        write_raw_bids(
            raw=self.raw,
            bids_path=bids_path,
            overwrite=overwrite,
            verbose=True,
            allow_preload=True,
            format='EDF',
            acpc_aligned=True,
        )

        if hasattr(self, '_ch_index_to_type'):
            self._add_electrode_type_to_tsvs(bids_path, verbose=verbose)

        self.bids_path = bids_path

        # Point the in-memory RawArray at the written file so
        # save_derivative can parse BIDS entities from inst.filenames.
        written_fpath = bids_path.copy().update(extension='.edf').fpath
        self.raw._filenames = [str(written_fpath)]

        # Standardise the auto-generated events.tsv
        self._reformat_events_tsv()

        # only need word-level for sentence task, main event tsv handles
        # word level in single-word tasks
        if self.task == 'sentence':
            # Write word-level derivative (raw copy + word events)
            self._write_subevent_derivative(
                'words',
                overwrite=overwrite,
                verbose=verbose,
            )

        # Write phoneme-level derivative (raw copy + phoneme events)
        self._write_subevent_derivative(
            'phones',
            overwrite=overwrite,
            verbose=verbose,
        )

        # Channel map derivative
        if hasattr(self, 'channel_map') and self.channel_map is not None:
            self._convert_channel_map_to_bids(
                overwrite=overwrite, verbose=verbose,
            )

        # Anatomy
        self._convert_anat_to_bids(overwrite=overwrite, verbose=verbose)

        # Audio derivative
        if hasattr(self, 'audio_files') and self.audio_files is not None:
            self.save_to_derivative(
                data=self.audio_files,
                folder='audio',
                filename='microphone',
                file_type='wav',
                overwrite=overwrite,
                verbose=verbose,
            )

        return self.bids_path

    # ------------------------------------------------------------------
    # Anatomy
    # ------------------------------------------------------------------

    def _create_empty_sidecar(self, bids_path: BIDSPath, modality: str):
        """Create a minimal JSON sidecar placeholder for an anatomical image."""
        sidecar_path = bids_path.fpath.parent / f"{bids_path.basename}.json"
        metadata = {
            "Modality": "MR" if modality == "T1w" else "CT",
            "Description": (
                f"Placeholder metadata for {modality} image. "
                "Full metadata including fiducials will be added after "
                "RAS coordinate processing."
            ),
            "GeneratedBy": [{
                "Name": "ECoG BIDS Conversion Tool",
                "Description": (
                    f"Temporary placeholder for {modality} metadata"
                ),
            }],
        }
        with open(sidecar_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _convert_anat_to_bids(self, overwrite=True, verbose=True):
        """Locate T1w/CT sources and write to ``anat/`` with placeholder sidecars."""
        possible_dirs = [
            self.recon_path / self.recon_subject / 'elec_recon',
            self.recon_path / self.recon_subject / 'mri',
        ]

        subject_anat_source_dir = None
        for dir_path in possible_dirs:
            if dir_path.is_dir():
                subject_anat_source_dir = dir_path
                if verbose:
                    print(f"Found anatomical directory: {dir_path}")
                break

        if subject_anat_source_dir is None:
            if verbose:
                print(
                    f"Could not find anatomical directory for subject "
                    f"{self.recon_subject} in any of: "
                    f"{[str(d) for d in possible_dirs]}"
                )
            return

        # --- T1w ---
        t1w_bids_path = BIDSPath(
            subject=self.subject,
            datatype='anat',
            suffix='T1w',
            root=self.bids_root,
        )

        t1w_source_file = subject_anat_source_dir / 'T1.nii.gz'
        if t1w_source_file.exists():
            if verbose:
                print(
                    f"Writing T1w anatomical data to: {t1w_bids_path.fpath}"
                )
            try:
                write_anat(
                    image=str(t1w_source_file),
                    bids_path=t1w_bids_path,
                    overwrite=overwrite,
                    verbose=verbose,
                )
                self._create_empty_sidecar(t1w_bids_path, 'T1w')
            except Exception as e:
                print(f"Error writing T1w anatomical data: {e}")
        else:
            import nibabel as nib
            subject_anat_source_dir = (
                self.recon_path / self.recon_subject / 'mri'
            )
            native_mgz = subject_anat_source_dir / 'native.mgz'
            temp_file = None
            try:
                img = nib.load(str(native_mgz))
                temp_dir = t1w_bids_path.fpath.parent
                temp_file = temp_dir / 'temp_mri.nii.gz'
                temp_dir.mkdir(parents=True, exist_ok=True)
                nib.save(img, str(temp_file))
                write_anat(
                    image=str(temp_file),
                    bids_path=t1w_bids_path,
                    overwrite=overwrite,
                    verbose=verbose,
                )
                self._create_empty_sidecar(t1w_bids_path, 'T1w')
            except Exception as e:
                print(f"Error writing T1w anatomical data: {e}")
            finally:
                if temp_file is not None and temp_file.exists():
                    temp_file.unlink()

        # flip CT orientation upside down
        

        # --- CT ---
        ct_source_file = subject_anat_source_dir / 'postimpRaw.nii.gz'
        if ct_source_file.exists():
            ct_bids_path = BIDSPath(
                subject=self.subject,
                datatype='anat',
                suffix='ct',
                root=self.bids_root,
                check=False,
            )
            if verbose:
                print(
                    f"Processing CT anatomical data to: "
                    f"{ct_bids_path.fpath}"
                )

            import shutil
            temp_dir = ct_bids_path.fpath.parent
            temp_file = temp_dir / 'temp_ct.nii'

            try:
                shutil.copy2(ct_source_file, temp_file)
                write_anat(
                    image=str(temp_file),
                    bids_path=ct_bids_path,
                    overwrite=overwrite,
                    verbose=verbose,
                )
                self._create_empty_sidecar(ct_bids_path, 'CT')
                if verbose:
                    print(
                        f"Successfully processed CT scan to "
                        f"{ct_bids_path.fpath}"
                    )
            except Exception as e:
                print(f"Error writing CT anatomical data: {e}")
            finally:
                if temp_file.exists():
                    temp_file.unlink()
        else:
            if verbose:
                print(
                    f"CT file (postimpRaw.nii.gz) not found: "
                    f"{ct_source_file}. Skipping CT conversion."
                )

    # ------------------------------------------------------------------
    # Channel map derivative
    # ------------------------------------------------------------------

    def _convert_channel_map_to_bids(self, overwrite: bool, verbose: bool):
        """Serialize the channel map to a TSV derivative."""
        if self.bids_path is None:
            raise ValueError(
                "BIDS path not set; call convert_to_bids() first"
            )

        if self.channel_map is None:
            if verbose:
                print(
                    "No channel map available; "
                    "skipping channel map derivative."
                )
            return None

        channel_map = np.asarray(self.channel_map)
        if channel_map.ndim != 2:
            raise ValueError(
                f"Expected a 2D channel map array, "
                f"got shape {channel_map.shape}"
            )

        rows = []
        raw_ch_names = self.raw.ch_names if self.raw is not None else None
        for row_idx in range(channel_map.shape[0]):
            for col_idx in range(channel_map.shape[1]):
                ch_num = channel_map[row_idx, col_idx]
                if pd.isna(ch_num):
                    continue
                ch_idx = int(ch_num)
                if raw_ch_names is None:
                    ch_name = str(int(ch_num))
                elif 0 <= ch_idx < len(raw_ch_names):
                    ch_name = raw_ch_names[ch_idx]
                else:
                    raise ValueError(
                        f"Channel map references channel number "
                        f"{int(ch_num)}, which is out of range for "
                        "the raw object"
                    )
                ch_type = (
                    self._ch_index_to_type.get(int(ch_num), "micro")
                    if hasattr(self, "_ch_index_to_type")
                    else "micro"
                )
                rows.append({
                    "row": row_idx,
                    "col": col_idx,
                    "channel_number": int(ch_num),
                    "name": ch_name,
                    "electrode_type": ch_type,
                })

        if not rows:
            if verbose:
                print(
                    "Channel map does not contain any recording channels; "
                    "skipping derivative write."
                )
            return None

        channel_map_df = pd.DataFrame(rows)
        channel_map_df = channel_map_df.sort_values(
            ["channel_number", "row", "col"],
        ).reset_index(drop=True)

        derivative_path = self.save_to_derivative(
            data=channel_map_df,
            folder="channelMap",
            filename="channelMap",
            file_type="tsv",
            description=None,
            overwrite=overwrite,
            verbose=verbose,
        )

        if verbose:
            print(
                f"Saved channel map derivative to: {derivative_path.fpath}"
            )

        return derivative_path

    # ------------------------------------------------------------------
    # In-place channel renaming on existing BIDS output
    # ------------------------------------------------------------------

    @staticmethod
    def _rename_edf_channels_by_index(edf_path, index_to_name, verbose=True):
        """Rename EDF channels by position index, ignoring current labels."""
        edf_path = Path(edf_path)
        if not edf_path.exists():
            if verbose:
                print(f"EDF not found, skipping: {edf_path}")
            return

        with open(edf_path, 'r+b') as f:
            f.seek(252)
            ns = int(f.read(4).decode('ascii').strip())

            for i in range(ns):
                if i not in index_to_name:
                    continue
                offset = 256 + i * 16
                new_label = index_to_name[i].ljust(16)[:16]
                f.seek(offset)
                f.write(new_label.encode('ascii'))

        if verbose:
            print(f"Updated EDF channel labels: {edf_path}")

    @staticmethod
    def _rename_tsv_name_by_index(
        tsv_path, index_to_name, index_to_type=None, verbose=True,
    ):
        """Rename the ``name`` column by row index, ignoring current values.

        If *index_to_type* is provided, an ``electrode_type`` column is
        also written (or updated) using the same positional mapping.
        """
        tsv_path = Path(tsv_path)
        if not tsv_path.exists():
            if verbose:
                print(f"TSV not found, skipping: {tsv_path}")
            return

        df = pd.read_csv(tsv_path, sep='\t')
        if 'name' not in df.columns:
            if verbose:
                print(f"No 'name' column in {tsv_path}, skipping")
            return

        df['name'] = [
            index_to_name.get(i, df.at[i, 'name'])
            for i in range(len(df))
        ]
        if index_to_type is not None:
            df['electrode_type'] = [
                index_to_type.get(i, "micro")
                for i in range(len(df))
            ]
        df.to_csv(tsv_path, sep='\t', index=False)

        if verbose:
            print(f"Updated TSV channel names: {tsv_path}")

    @staticmethod
    def _rename_chanmap_tsv(
        tsv_path, index_to_name, index_to_type=None, verbose=True,
    ):
        """Rename the ``name`` column using the ``channel_number`` column as key.

        If *index_to_type* is provided, an ``electrode_type`` column is
        also written (or updated) using ``channel_number`` as the lookup
        key.
        """
        tsv_path = Path(tsv_path)
        if not tsv_path.exists():
            if verbose:
                print(f"TSV not found, skipping: {tsv_path}")
            return

        df = pd.read_csv(tsv_path, sep='\t')
        if 'name' not in df.columns or 'channel_number' not in df.columns:
            if verbose:
                print(f"Missing columns in {tsv_path}, skipping")
            return

        df['name'] = df['channel_number'].apply(
            lambda x: index_to_name.get(int(x), str(int(x))),
        )
        if index_to_type is not None:
            df['electrode_type'] = df['channel_number'].apply(
                lambda x: index_to_type.get(int(x), "micro"),
            )
        df.to_csv(tsv_path, sep='\t', index=False)

        if verbose:
            print(f"Updated channel map TSV: {tsv_path}")

    @staticmethod
    def _strip_acq_run(filename):
        """Remove ``_acq-*`` and ``_run-*`` entities from a BIDS filename."""
        return re.sub(r'_(?:acq|run)-[^_.]+', '', filename)

    def update_bids_channel_names(self, verbose=True):
        """Rename channels in an existing BIDS dataset to ``R{row}-C{col}`` format.

        Assigns names based on each channel's **position index** in the file
        (0-indexed), making the result independent of whatever the current
        names happen to be.  Also strips ``_acq-*`` and ``_run-*`` entities
        from BIDS filenames.
        """
        subj_dir = os.path.join(
            self.source_path, f'sub-{self.subject}',
        )
        standalone_map = os.path.join(
            subj_dir, f'sub-{self.subject}_channel_map.npy',
        )
        if os.path.exists(standalone_map):
            self.channel_map = np.load(standalone_map)
        else:
            raw_path = os.path.join(
                subj_dir, f'sub-{self.subject}_raw.h5',
            )
            with h5py.File(raw_path, 'r') as f:
                self.channel_map = f['channel_map'][()]

        self._ch_index_to_name = self._build_channel_index_to_name()
        subject_id = self.subject.lstrip('sub-')

        # -- rename channels by position index --

        edf_files = list(self.bids_root.rglob(
            f'sub-{subject_id}*_ieeg.edf',
        ))
        for f in edf_files:
            self._rename_edf_channels_by_index(
                f, self._ch_index_to_name, verbose,
            )

        index_to_type = getattr(self, '_ch_index_to_type', None)

        for suffix in ('channels', 'electrodes'):
            tsv_files = list(self.bids_root.rglob(
                f'sub-{subject_id}*_{suffix}.tsv',
            ))
            for f in tsv_files:
                self._rename_tsv_name_by_index(
                    f, self._ch_index_to_name, index_to_type, verbose,
                )

        chanmap_tsvs = list(self.bids_root.rglob(
            f'sub-{subject_id}*channelMap*.tsv',
        ))
        for f in chanmap_tsvs:
            self._rename_chanmap_tsv(
                f, self._ch_index_to_name, index_to_type, verbose,
            )

        # -- strip acq/run entities from filenames --

        subject_files = sorted(
            self.bids_root.rglob(f'sub-{subject_id}*'),
            reverse=True,
        )
        for f in subject_files:
            if not f.is_file():
                continue
            new_name = self._strip_acq_run(f.name)
            if new_name != f.name:
                new_path = f.parent / new_name
                f.replace(new_path)
                if verbose:
                    print(f"Renamed: {f.name} -> {new_name}")

        # scans_tsv = (
        #     self.bids_root / f'sub-{subject_id}'
        #     / f'sub-{subject_id}_scans.tsv'
        # )
        # if scans_tsv.exists():
        #     df = pd.read_csv(scans_tsv, sep='\t')
        #     if 'filename' in df.columns:
        #         df['filename'] = df['filename'].apply(
        #             self._strip_acq_run,
        #         )
        #         df.to_csv(scans_tsv, sep='\t', index=False)
        #         if verbose:
        #             print(f"Updated scans.tsv paths: {scans_tsv}")

        if verbose:
            print("Channel name update complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path='config',
            config_name='bids_writer')
def main(cfg: DictConfig) -> None:
    """Run the full Intan RHD -> BIDS conversion pipeline."""
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration:\n{OmegaConf.to_yaml(cfg)}')

    try:
        from dataloaders import rhdLoader
    except ImportError:
        from .dataloaders import rhdLoader

    subject = cfg.subject
    task = cfg.task
    source_path = (
        Path(cfg.source_path).expanduser()
        if cfg.source_path is not None else None
    )
    recon_path = (
        Path(cfg.recon_path).expanduser()
        if cfg.recon_path is not None else None
    )
    
    fileIDs = tuple(cfg.fileIDs) if cfg.fileIDs is not None else None
    array_type = cfg.array_type
    num_arrays = int(cfg.num_arrays)

    missing_data_idx = _parse_missing_data_idx(cfg.missing_data_idx, cfg.missing_data_idx_type)

    if task == 'lexical':
        bids_root = Path.home() / 'Box' / 'CoganLab' / 'BIDS_1.0_Lexical_\u00b5ECoG' / 'BIDS'
    elif task == 'phoneme':
        bids_root = Path.home() / 'Box' / 'CoganLab' / 'BIDS_1.0_Phoneme_Sequence_uECoG' / 'BIDS'
    elif task == 'sentence':
        bids_root = Path.home() / 'Box' / 'CoganLab' / 'BIDS_1.0_Sentence_\u00b5ECoG' / 'BIDS'
    else:
        raise ValueError(f'Invalid task: {task}')

    loader = rhdLoader(
        subject, source_path, fileIDs=fileIDs, array_type=array_type,
        num_arrays=num_arrays,
        missing_data_idx=missing_data_idx,
    )

    # standard data loading pipeline
    # loader.load_data(use_ext_mic=cfg.use_ext_mic)
    # loader.make_cue_events(
    #     use_ext_trigger=cfg.use_ext_trigger, use_ext_mic=cfg.use_ext_mic,
    # )
    # loader.run_mfa(task_name=TASK2MFA[task], denoise_mic=(not cfg.use_ext_mic))

    # # update impedance and bad channels
    # loader.update_impedance()

    bids_converter = BIDSConverter(
        source_path=loader.out_dir,
        subject=subject,
        task=task,
        bids_root=bids_root,
        recon_path=recon_path,
    )
    bids_converter.convert_to_bids()

    # rename channels in existing BIDS output to R{row}-C{col} format
    # bids_converter.update_bids_channel_names()


if __name__ == '__main__':
    main()
