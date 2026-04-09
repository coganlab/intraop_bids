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
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids, write_anat

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

        channel_names = [f'{i}' for i in range(len(raw_data))]
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.fs,
            ch_types='ecog',
        )
        info['bads'] = [str(ch) for ch in raw['bad_channels'][()]]
        raw.close()

        raw_data = raw_data.astype(np.float32)
        raw = mne.io.RawArray(data=raw_data * 1e-6, info=info)
        del raw_data
        import gc
        gc.collect()

        return raw

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
            montage = self.make_custom_montage(self.channel_map)
            self.raw.set_montage(montage)
            return self.raw
        try:
            montage = self.load_ras_montage(ras_file)
            self.raw.set_montage(montage)
        except Exception as e:
            print(f"Error applying RAS montage: {e}")
            print("Falling back to custom montage...")
            montage = self.make_custom_montage(self.channel_map)
            self.raw.set_montage(montage)

        return self.raw

    @staticmethod
    def load_ras_montage(ras_file):
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
        num_map = {old: i + 1 for i, old in enumerate(unique_sorted)}
        ch_names = [str(num_map[n]) for n in orig_nums]
        pos = df[['x', 'y', 'z']].values

        montage = make_dig_montage(
            ch_pos=dict(zip(ch_names, pos)),
            coord_frame='ras',
        )

        for ch, hemi in zip(montage.dig, df['hemisphere']):
            if ch['kind'] == FIFF.FIFFV_POINT_EEG:
                ch['hemisphere'] = hemi.upper()

        return montage

    @staticmethod
    def make_custom_montage(channel_map):
        """Create a synthetic 2D-grid montage as a placeholder."""
        from mne.channels import Layout, make_dig_montage

        n_rows, n_cols = channel_map.shape
        pos, names = [], []

        for i in range(n_rows):
            for j in range(n_cols):
                if not np.isnan(channel_map[i, j]):
                    x = j / n_cols
                    y = (n_rows - i - 1) / n_rows
                    width = 1 / n_cols
                    height = 1 / n_rows
                    pos.append([x, y, width, height])
                    names.append(f'{int(channel_map[i, j])}')

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
        stim_file = os.path.join(subj_dir, "mfa_stim_words.txt")
        resp_file = os.path.join(subj_dir, "mfa_resp_words.txt")

        onsets_sec, durations_sec, descriptions = [], [], []

        if os.path.exists(stim_file):
            stim_words = np.loadtxt(stim_file, dtype=str)
            for row in stim_words:
                onset_sec, offset_sec, word = row[0], row[1], row[2]
                onsets_sec.append(float(onset_sec))
                durations_sec.append(
                    max(0.0, float(offset_sec) - float(onset_sec))
                )
                descriptions.append(f"stimulus/{word}")

        if os.path.exists(resp_file):
            resp_words = np.loadtxt(resp_file, dtype=str)
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

        subj_dir = self.source_path / f'sub-{self.subject}'
        stim_phon_file = subj_dir / 'mfa_stim_phones.txt'
        resp_phon_file = subj_dir / 'mfa_resp_phones.txt'

        if not stim_phon_file.exists() or not resp_phon_file.exists():
            logger.warning(
                "Phoneme annotation files not found in %s; "
                "skipping phoneme-level derivative events.",
                subj_dir,
            )
            return

        sfreq = self.raw.info['sfreq']
        n_phons = NPHONS.get(self.task, 5)

        stim_df = phon_txt2df(
            stim_phon_file, 'stimulus', self.subject,
            n_phons, sfreq, self.task,
        )
        resp_df = phon_txt2df(
            resp_phon_file, 'response', self.subject,
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

        self.bids_path = bids_path

        # Re-read so self.raw has valid filenames for save_derivative
        self.raw = read_raw_bids(bids_path, verbose='ERROR')

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
                rows.append({
                    "row": row_idx,
                    "col": col_idx,
                    "channel_number": int(ch_num),
                    "name": ch_name,
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(
    source_path: Path,
    bids_root: Path,
    subject: str,
    task: str,
    fileIDs: Optional[list] = None,
    array_type: Optional[str] = None,
    recon_path: Optional[Path] = None,
):
    """Run the full Intan RHD -> BIDS conversion pipeline."""
    try:
        from dataloaders import rhdLoader
    except ImportError:
        from .dataloaders import rhdLoader

    loader = rhdLoader(
        subject, source_path, fileIDs=fileIDs, array_type=array_type,
    )
    loader.update_impedance()
    # loader.load_data()
    # loader.make_cue_events()
    # loader.run_mfa(task_name=TASK2MFA[task])

    bids_converter = BIDSConverter(
        source_path=loader.out_dir,
        subject=subject,
        task=task,
        bids_root=bids_root,
        recon_path=recon_path,
    )

    bids_converter.convert_to_bids()


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
        "--bids-root",
        default=(
            Path.home() / "Box" / "CoganLab"
            / "BIDS_1.0_Lexical_\u00b5ECoG" / "BIDS"
        ),
        type=Path,
        help='Root directory to save the BIDS dataset',
    )
    parser.add_argument(
        "--subject",
        default=None,
        help='Subject identifier (e.g., S41)',
    )
    parser.add_argument(
        "--fileIDs",
        nargs='+',
        default=None,
        help=(
            'List of file IDs to process indicating the RHD files to process '
            'when ordered alphabetically in the source data directory (e.g. '
            'for the 5th through 10th files, use --fileIDs 5 6 7 8 9 10)'
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
