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

import h5py
import mne
import numpy as np
import os
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids, write_anat

logger = logging.getLogger(__name__)

TASK2MFA = {
    "lexical": "lexical_repeat_intraop",
    "phoneme": "phoneme_sequencing",
}

NPHONS = {
    "lexical": 5,
    "phoneme": 3,
}

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


def remove_orphan_stimuli(events_df, response_window_sec=4.0):
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


def phon_txt2df(txt_path, trial_type, subject, n_phons, fs, task):
    """Convert a phoneme-level MFA text file to a BIDS events DataFrame."""
    df = pd.read_csv(txt_path, sep='\t', names=['onset', 'offset', 'phoneme'])
    df['duration'] = df['offset'] - df['onset']

    if task == 'lexical':
        df['value'] = df['phoneme'].apply(remove_arpabet_stress)
    else:
        df['value'] = df['phoneme'].apply(
            lambda x: PS2ARPA.get(
                remove_arpabet_stress(x), remove_arpabet_stress(x)
            )
        )

    df['subject'] = subject
    df['sample'] = (df['onset'] * fs).astype(int)
    df['trial'] = (np.arange(len(df)) // n_phons) + 1
    position = (np.arange(len(df)) % n_phons) + 1
    df['trial_type'] = [f'{trial_type}/{p}' for p in position]
    return df[EVENTS_COL_ORDER]


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
        info['bads'] = [
            self._ch_index_to_name.get(int(ch), str(ch))
            for ch in raw['bad_channels'][()]
        ]
        raw.close()

        raw_data = raw_data.astype(np.float32)
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
            f"{self.subject}",
            "elec_recon",
            f"{self.subject}_elec_locations_RAS_brainshifted.txt",
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
            for row in stim_words:
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
            for row in resp_words:
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
        events_df = remove_orphan_stimuli(events_df)
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
    # Phoneme-level derivative
    # ------------------------------------------------------------------

    def _write_phoneme_derivative(self, overwrite=True, verbose=True):
        """Copy raw data and write phoneme-level events to derivatives/phonemeLevel/.

        The derivative folder is treated as its own BIDSLayout by downstream
        preprocessing code, so the raw data file must be present alongside
        the events.
        """
        from bids import BIDSLayout
        from ieeg.io import save_derivative

        bids_layout = BIDSLayout(
            root=str(self.bids_root), derivatives=True,
        )
        save_derivative(self.raw, bids_layout, 'phonemeLevel', overwrite)

        if hasattr(self, '_ch_index_to_type'):
            deriv_path = BIDSPath(
                root=self.bids_root / 'derivatives' / 'phonemeLevel',
                subject=self.subject,
                task=self.task,
                datatype='ieeg',
                suffix='ieeg',
                check=False,
            )
            self._add_electrode_type_to_tsvs(deriv_path, verbose=verbose)

        subj_dir = self.source_path / f'sub-{self.subject}'
        stim_phon_file = [subj_dir / 'mfa_adj_stim_phones.txt', subj_dir / 'mfa_stim_phones.txt']
        resp_phon_file = [subj_dir / 'mfa_adj_resp_phones.txt', subj_dir / 'mfa_resp_phones.txt']

        if stim_phon_file[0].exists():
            stim_fname = stim_phon_file[0]
        elif stim_phon_file[1].exists():
            stim_fname = stim_phon_file[1]
        else:
            logger.error(f"Stimulus phoneme file not found in {subj_dir}")
            raise FileNotFoundError(f"Stimulus phoneme file not found in {subj_dir}")
        
        if resp_phon_file[0].exists():
            resp_fname = resp_phon_file[0]
        elif resp_phon_file[1].exists():
            resp_fname = resp_phon_file[1]
        else:
            logger.error(f"Response phoneme file not found in {subj_dir}")
            raise FileNotFoundError(f"Response phoneme file not found in {subj_dir}")

        sfreq = self.raw.info['sfreq']
        n_phons = NPHONS.get(self.task, 5)

        stim_df = phon_txt2df(
            stim_fname, 'stimulus', self.subject,
            n_phons, sfreq, self.task,
        )
        resp_df = phon_txt2df(
            resp_fname, 'response', self.subject,
            n_phons, sfreq, self.task,
        )

        events_df = (
            pd.concat([stim_df, resp_df])
            .sort_values('onset')
            .reset_index(drop=True)
        )
        events_df = remove_orphan_stimuli(events_df)
        events_df['trial'] = np.ceil(
            np.arange(1, len(events_df) + 1) / (n_phons * 2)
        ).astype(int)

        events_path = BIDSPath(
            root=self.bids_root / 'derivatives' / 'phonemeLevel',
            subject=self.subject,
            task=self.task,
            datatype='ieeg',
            suffix='events',
            description='phonemeLevel',
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
                "Wrote phoneme-level events to %s", events_path.fpath,
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

        # Write phoneme-level derivative (raw copy + phoneme events)
        self._write_phoneme_derivative(
            overwrite=overwrite, verbose=verbose,
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
            self.recon_path / self.subject / 'elec_recon',
            self.recon_path / self.subject / 'mri',
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
                    f"{self.subject} in any of: "
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
                self.recon_path / self.subject / 'mri'
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

def main(
    source_path: Path,
    subject: str,
    task: str,
    fileIDs: Optional[tuple] = None,
    array_type: Optional[str] = None,
    num_arrays: int = 1,
    recon_path: Optional[Path] = None,
):
    """Run the full Intan RHD -> BIDS conversion pipeline."""
    try:
        from dataloaders import rhdLoader
    except ImportError:
        from .dataloaders import rhdLoader

    if task == 'lexical':
        bids_root = Path.home() / 'Box' / 'CoganLab' / 'BIDS_1.0_Lexical_\u00b5ECoG' / 'BIDS'
    elif task == 'phoneme':
        bids_root = Path.home() / 'Box' / 'CoganLab' / 'BIDS_1.0_Phoneme_Sequence_uECoG' / 'BIDS'
    else:
        raise ValueError(f'Invalid task: {task}')

    loader = rhdLoader(
        subject, source_path, fileIDs=fileIDs, array_type=array_type,
        num_arrays=num_arrays,
    )

    # # standard data loading pipeline
    # loader.load_data()
    # loader.make_cue_events()
    # loader.run_mfa(task_name=TASK2MFA[task])

    # update impedance and bad channels
    loader.update_impedance() 

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-path",
        default=None,
        help=(
            'Directory containing the RHD source data files. If not provided, '
            'the user will be prompted to select a directory via GUI. This '
            'directory should also contain the trialInfo file from the task '
            'computer.'
        ),
    )
    parser.add_argument(
        "--subject",
        default=None,
        help='Subject identifier (e.g., S41)',
    )
    parser.add_argument(
        "--fileIDs",
        nargs=2,
        type=int,
        default=None,
        metavar=('START', 'END'),
        help=(
            'Inclusive start and end (1-indexed) of the contiguous range of '
            'RHD files to process when ordered alphabetically in the source '
            'data directory (e.g. for the 5th through 10th files, use '
            '--fileIDs 5 10)'
        ),
    )
    parser.add_argument(
        "--array-type",
        default=None,
        help=(
            'Type of electrode array '
            '(128-strip, 256-grid, 256-strip, 1024-grid, hybrid-strip)'
        ),
    )
    parser.add_argument(
        "--num-arrays",
        type=int,
        default=1,
        help=(
            'Number of identical arrays used (default: 1). '
            'When > 1, the channel map is vertically stacked with '
            'offset channel indices for each copy.'
        ),
    )
    parser.add_argument(
        "--recon-path",
        default=Path.home() / "Box" / "ECoG_Recon",
        type=Path,
        help=(
            'Parent directory containing subject anatomical subdirectories '
            '(e.g., ECoG_Recon/)'
        ),
    )
    parser.add_argument(
        "--task",
        default="lexical",
        help='Name of the task (default: lexical)',
    )

    args = parser.parse_args()
    main(**vars(args))
