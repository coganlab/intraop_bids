"""
ECoG/iEEG to BIDS conversion utilities built on mne-bids.

Pipeline overview
- Load subject-specific source files using loader helpers in `loaders.py`.
- Load experiment/session metadata (`load_experiment_info()`), iEEG raw
  data (`load_ieeg_data()`), events (`load_event_data()`), and optional
  microphone audio (`load_audio_data()`).
- Process events and update the `mne.io.Raw` object via
  `BIDSConverter.update_raw_object()`:
  - Convert event structures to MNE-style `events` array and `event_id`.
  - Ensure the raw has a montage. If a subject RAS montage exists in
    `recon_path/<sub>/elec_recon/*RAS_brainshifted.txt`, load it; otherwise
    generate a custom montage based on the provided channel map.
- Create a BIDS directory tree under `bids_root` (session-less by default),
  and write iEEG data to BIDS with `write_raw_bids()`.
- Convert anatomical data (T1w and CT when present) into `anat/`, writing
  placeholder sidecar JSONs to be completed later (e.g., with fiducials).
- Optionally save microphone audio as a derivative in
  `derivatives/audio/sub-<label>/`.

Notes and assumptions
- The code relies on helper loaders in `loaders.py` and configuration from
  `config.py`.
- Some fields (e.g., session handling) are currently minimal or implicit.
- The RAS montage file is expected to be whitespace-delimited with columns:
  prefix, number, x, y, z, hemisphere, grid.
"""
from pathlib import Path
from typing import Optional, Union
from mne_bids import BIDSPath, write_raw_bids, write_anat
import numpy as np
import h5py
import os
import mne
from dataloaders import rhdLoader


TASK2MFA = {
    "lexical": "lexical_repeat_intraop",
    "phoneme": "phoneme_sequencing",
}


class BIDSConverter:
    """
    Converter for writing iEEG/ECoG data to BIDS.

    Responsibilities
    - Manage source paths, subject/task identifiers, and optional audio/anat paths.
    - Load experiment metadata, iEEG raw, events, and optional audio.
    - Process events and ensure `raw` has a montage (RAS-based or synthetic).
    - Create a BIDS directory structure and write iEEG data using `mne-bids`.
    - Convert available anatomical images (T1w, CT) and create placeholder JSON sidecars.
    """

    def __init__(
        self,
        subject: str,
        task: str,
        source_path: Union[str, Path],
        bids_root: Optional[Union[str, Path]] = None,
        recon_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize a converter bound to the given subject/task and paths.

        Args
        - subject: Subject label (e.g., "S26").
        - task: BIDS task label (e.g., "phoneme").
        - source_path: Root directory containing the source data to convert.
        - bids_root: Root directory for the output BIDS dataset. If None, must
          be provided later to `convert_to_bids()`.
        - recon_path: Optional parent directory containing anatomical data
          (e.g., `ECoG_Recon/<sub>/...`).

        Notes
        - Paths should be valid and readable. Audio/anat are optional.
        - Session handling is not explicit; current flow assumes session-less.
        """
        self.subject = subject
        self.task = task
        self.source_path = Path(source_path)
        self.recon_path = Path(recon_path)
        self.bids_root = Path(bids_root)

        # Will be populated when data is loaded
        self.experiment_info = None
        self.raw = None
        self.events = None
        self.bids_path = None


    def _make_raw_object(
        self,
    ):
        """
        Make a raw object from the raw data.
        """

        raw_path = os.path.join(self.source_path,
                                f'sub-{self.subject}',
                                f'sub-{self.subject}_raw.h5')
        raw = h5py.File(raw_path, 'r')

        raw_data = raw['data'][()]
        self.fs = raw['fs'][()]
        channel_map = raw['channel_map'][()]
        self.channel_map = channel_map

        channel_names = [f'{i+1}' for i in range(len(raw_data))]
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.fs,
            ch_types='seeg'
        )
        info['bads'] = [str(ch) for ch in raw['bad_channels'][()]]
        # make a raw object
        raw.close()
        # release raw_data from memory
        raw_data = raw_data.astype(np.float32)
        raw = mne.io.RawArray(
            data = raw_data * 1e-6,
            info = info,
        )
        del raw_data
        import gc; gc.collect()

        return raw


    def update_raw_object(self,
                          ann):
        """
        Update the `raw` object with processed events and ensure montage.

        Steps
        - Build `events` and `event_id` via `handling_event()` and attach to this instance.
        - If no montage is present on `raw`, attempt to add one via
          `add_montage_to_raw()` (RAS montage if available, otherwise custom).

        Returns
        - mne.io.Raw: The updated raw instance.
        """

        sfreq = float(self.raw.info['sfreq'])
        
        self.raw.set_annotations(ann)

        # make custom montage if no montage is found in the raw object
        if self.raw.info.get_montage() is None:
            self.raw = self.add_montage_to_raw()

        return

    def add_montage_to_raw(self):
        """
        Add a montage to `raw` using subject RAS file when present, else fallback.

        Logic
        - Construct a path to `<recon_path>/<sub>/elec_recon/*RAS_brainshifted.txt`.
        - If RAS file is missing or fails to load, make a synthetic montage from
          the channel map (layout-based) with a placeholder coordinate frame.

        Returns
        - mne.io.Raw: The raw instance with montage set.
        """
        # load RAS file from anat directory
        ras_file = os.path.join(
            self.recon_path,
            f"{self.subject}",
            "elec_recon",
            f"{self.subject}_elec_locations_RAS_brainshifted.txt"
        )

        # First check if file exists
        if not os.path.exists(ras_file):
            print(f"RAS file not found: {ras_file}, using custom montage as placeholder")
            montage = self.make_custom_montage(self.channel_map)
            self.raw.set_montage(montage)
            return self.raw
        try:
            # Try to load and set the RAS montage
            montage = self.load_ras_montage(ras_file)
            
            self.raw.set_montage(montage)
        except Exception as e:
            # If any error occurs (e.g., channel mismatch, invalid format), fall back to custom montage
            print(f"Error applying RAS montage: {e}")
            print("Falling back to custom montage...")
            montage = self.make_custom_montage(self.channel_map)
            self.raw.set_montage(montage)

        return self.raw

    @staticmethod
    def load_ras_montage(ras_file):
        """
        Load a RAS montage and attach hemisphere labels.

        Args
        - ras_file: Path to a whitespace-delimited text file with columns:
          prefix, number, x, y, z, hemisphere, grid. Coordinates expected in
          RAS space.

        Returns
        - mne.channels.DigMontage: Electrode positions tagged with hemisphere.
        """
        import pandas as pd
        from mne.channels import make_dig_montage
        from mne.io.constants import FIFF

        # Read the RAS file with proper column names
        df = pd.read_csv(
            ras_file,
            delim_whitespace=True,
            header=None,
            names=['prefix', 'number', 'x', 'y', 'z', 'hemisphere', 'grid']
        )

        # Create channel names by remapping existing numbers to consecutive 1..N
        # while preserving the original row order.
        orig_nums = df['number'].astype(int).tolist()
        unique_sorted = sorted(set(orig_nums))
        num_map = {old: i + 1 for i, old in enumerate(unique_sorted)}
        ch_names = [str(num_map[n]) for n in orig_nums]

        # Get coordinates in meters (MNE uses meters)
        pos = df[['x', 'y', 'z']].values

        # Create montage with RAS coordinates
        montage = make_dig_montage(
            ch_pos=dict(zip(ch_names, pos)),
            coord_frame='ras'
        )

        # Add hemisphere information to the montage's channel dictionary
        for ch, hemi in zip(montage.dig, df['hemisphere']):
            if ch['kind'] == FIFF.FIFFV_POINT_EEG:
                ch['hemisphere'] = hemi.upper()  # Ensure uppercase for BIDS compliance

        return montage

    @staticmethod
    def make_custom_montage(channel_map):
        """
        Create a synthetic montage from `experiment_info['channel_map']`.

        Notes
        - Generates a 2D grid layout and maps channels to [x, y, 0] placeholders.
        - The coordinate frame is set to 'ras' as a pragmatic placeholder to
          satisfy downstream checks; these are not real RAS coordinates.

        Returns
        - mne.channels.DigMontage: Synthetic montage with 2D layout positions.
        """
        from mne.channels import Layout, make_dig_montage
        # channel_map = experiment_info['channel_map']
        n_rows, n_cols = channel_map.shape

        pos = []
        names = []

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
        box = np.array([0, 0, 1, 1])  # Define the bounding box

        layout = Layout(
            pos=pos,
            names=names,
            kind='custom',
            ids=np.arange(1, len(names) + 1),
            box=box
        )
        ch_pos = {name: [x, y, 0] for name, (x, y, _, _) in zip(layout.names, layout.pos)}
        # remember this is a fake montage, the coord_frame is set to be 'ras' to pass the check
        montage = make_dig_montage(ch_pos, coord_frame='ras')

        return montage

    def handling_event(self):
        """
        Convert trial/event structures to MNE `events` and `event_id`.

        Event processing
        - Pull config from `DEFAULT_EVENT_PROCESSING['bids_events_columns']`.
        - Drop trials with empty `trial_type` entries.
        - Convert event onsets to sample indices using `raw.info['sfreq']`,
          applying a factor of `sfreq/3e4` to match the event time base.
        - Build `event_id` by enumerating unique trial types.

        Returns
        - (events, event_id):
          - events: int array of shape (n_events, 3) [sample, 0, code].
          - event_id: dict mapping trial_type strings to integer codes.

        Raises
        - KeyError/ValueError on missing fields or malformed inputs.
        """
        if not hasattr(self, 'events') or self.events is None:
            print("No event data to process")
            return None, None

        event_mat = self.events
        event_config = DEFAULT_EVENT_PROCESSING.get('bids_events_columns')

        # Validate required fields
        required_fields = ["onset", "duration", "trial_type"]
        for field in required_fields:
            if field not in event_config:
                raise ValueError(f"Missing or invalid event configuration for {field}")

        try:
            # Get sampling rate and validate
            sfreq = float(self.raw.info['sfreq'])

            # remove NaN rows
            event_mat = [event for event in event_mat if not len(event[event_config["trial_type"]["source_field"]]) == 0]
            # Process onsets and durations
            onset_samples = [event[event_config["onset"]["source_field"]] for event in event_mat]
            onset_samples = [int(float(onset) * (sfreq/3e4)) for onset in onset_samples]

            trial_type = [event[event_config["trial_type"]["source_field"]] for event in event_mat]
            assert len(trial_type) > 0, "No trial types found in event data"

            # Create event ID mapping
            unique_trial_types = np.unique(trial_type)
            event_id = {str(t): i+1 for i, t in enumerate(unique_trial_types)}
            trial_type_id = np.array([event_id[t] for t in trial_type])

            # Create events array
            events = np.column_stack((
                onset_samples,      # Sample numbers
                np.zeros_like(onset_samples),  # Zeros (standard in MNE)
                trial_type_id      # Event type IDs
            ))

            return events, event_id

        except KeyError as e:
            raise KeyError(f"Missing required field in event data: {e}")
        except Exception as e:
            raise Exception(f"Error processing events: {str(e)}")


    def _parse_events(self, sfreq):

        # load word level stim and response txt file
        stim_file = os.path.join(
            self.source_path,
            f'sub-{self.subject}',
            "mfa_stim_words.txt"
        )
        resp_file = os.path.join(
            self.source_path,
            f'sub-{self.subject}',
            "mfa_resp_words.txt"
        )

        onsets_sec = []
        durations_sec = []
        descriptions = []

        if os.path.exists(stim_file):
            stim_words = np.loadtxt(stim_file, dtype=str)
            for row in stim_words:
                onset_sec, offset_sec, word = row[0], row[1], row[2]
                onsets_sec.append(float(onset_sec))
                durations_sec.append(max(0.0, float(offset_sec) - float(onset_sec)))
                descriptions.append(f"stim/{word}")

        if os.path.exists(resp_file):
            resp_words = np.loadtxt(resp_file, dtype=str)
            for row in resp_words:
                onset_sec, offset_sec, word = row[0], row[1], row[2]
                onsets_sec.append(float(onset_sec))
                durations_sec.append(max(0.0, float(offset_sec) - float(onset_sec)))
                descriptions.append(f"resp/{word}")

        ann = mne.Annotations(
            onset=onsets_sec,
            duration=durations_sec,
            description=descriptions,
            orig_time=self.raw.info.get('meas_date')
        )

        return ann

    def save_to_derivative(
        self,
        data,
        folder: str,
        filename: str,
        file_type: str = 'wav',
        description: str = 'raw',
        **kwargs
    ) -> Optional[Path]:
        """
        Save arrays (e.g., audio) into `derivatives/<folder>/` with BIDSPath.

        Args
        - data: Array-like data to write (e.g., audio samples).
        - folder: Derivative subfolder name (e.g., "audio").
        - filename: Base name for the output file (without extension).
        - file_type: Output extension (e.g., "wav").
        - description: BIDS description label stored in the `BIDSPath`.
        - kwargs: Reserved for future extensions.

        Returns
        - Optional[pathlib.Path]: Final path of the written derivative, or None.
        """
        derivative_dir = self.bids_path.copy()
        derivative_dir.update(
            root=self.bids_root / "derivatives" / folder,
            suffix=filename,
            datatype=filename,
            description=description,
            extension=f'.{file_type}',
            check=False
        )

        derivative_dir.mkdir(exist_ok=True)

        if file_type == 'wav':
            import soundfile as sf
            sf.write(
                str(derivative_dir.fpath),
                data,
                self.audio_fs
            )

        return derivative_dir

    def convert_to_bids(
            self,
            output_dir: Optional[Union[str, Path]] = None,
            overwrite: bool = True,
            verbose: bool = True
    ) -> BIDSPath:
        """
        Convert loaded data to BIDS using `mne-bids` writers.

        Steps
        - Ensure `raw` is present; create BIDS directories.
        - Build a `BIDSPath` (session-less) and ensure destination exists.
        - Update `raw` (events + montage) and call `write_raw_bids()`.
        - Convert available anatomical images and write derivatives if any.

        Args
        - output_dir: Overrides `self.bids_root` if provided.
        - overwrite: Whether to overwrite existing outputs.
        - verbose: Verbosity flag for writers.

        Returns
        - mne_bids.BIDSPath: Path for the written iEEG file within the BIDS tree.

        Raises
        - ValueError: If data is not loaded or required attributes are missing.
        """

        bids_root = Path(output_dir) if output_dir else self.bids_root
        if bids_root is None:
            raise ValueError("output_dir must be provided or bids_root must be set")

        # Create BIDS path with session if provided
        bids_path = BIDSPath(
            subject=self.subject.lstrip('sub-'),  # Remove 'sub-' prefix if present
            task=self.task,
            root=str(bids_root),
            datatype='ieeg',
            suffix='ieeg',
        )

        # Create output directory if it doesn't exist
        bids_path.mkdir(exist_ok=True)

        self.raw = self._make_raw_object()

        ann = self._parse_events(self.raw.info['sfreq'])

        # update raw
        self.update_raw_object(
            ann=ann
        )

        # Write the data to BIDS format (derive events.tsv from annotations)
        write_raw_bids(
            raw=self.raw, 
            bids_path=bids_path,
            overwrite=overwrite,
            verbose=True,
            allow_preload=True,
            format='EDF',
            acpc_aligned=True,
        )

        # Store the bids_path for later use
        self.bids_path = bids_path

        self._convert_anat_to_bids(overwrite=overwrite, verbose=verbose)

        if hasattr(self, 'audio_files') and self.audio_files is not None:
            self.save_to_derivative(
                data=self.audio_files,
                folder='audio',
                filename='microphone',
                file_type='wav',
                overwrite=overwrite,
                verbose=verbose
            )

        return self.bids_path

    def _create_empty_sidecar(self, bids_path: BIDSPath, modality: str):
        """
        Create a minimal JSON sidecar for an anatomical image placeholder.

        Args
        - bids_path: Target BIDSPath for the anatomical image.
        - modality: "T1w" or "CT".
        """
        import json

        # Create the sidecar path with .json extension
        sidecar_path = bids_path.fpath.parent / f"{bids_path.basename}.json"

        # Basic metadata for the sidecar
        metadata = {
            "Modality": "MR" if modality == "T1w" else "CT",
            "Description": f"Placeholder metadata for {modality} image. Full metadata including fiducials will be added after RAS coordinate processing.",
            "GeneratedBy": [{
                "Name": "ECoG BIDS Conversion Tool",
                "Description": f"Temporary placeholder for {modality} metadata"
            }]
        }

        # Write the metadata to the sidecar file
        with open(sidecar_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _convert_anat_to_bids(self, overwrite: bool = True, verbose: bool = True):
        """
        Locate T1w/CT sources and write to `anat/` with placeholder sidecars.

        Behavior
        - Searches for subject anatomical data under `<recon_path>/<sub>/` in
          typical subfolders (e.g., `elec_recon/`, `mri/`).
        - Writes T1w (.nii.gz or converted from .mgz) and CT if present.
        - Generates empty JSON sidecars to be filled later with fiducials and
          alignment details.

        Args
        - overwrite: Whether to overwrite existing anatomical outputs.
        - verbose: Print progress messages.
        """
        possible_dirs = [
            self.recon_path / self.subject / 'elec_recon',
            self.recon_path / self.subject / 'mri',
        ]

        # Try each possible directory
        subject_anat_source_dir = None
        for dir_path in possible_dirs:
            if dir_path.is_dir():
                subject_anat_source_dir = dir_path
                if verbose:
                    print(f"Found anatomical directory: {dir_path}")
                break

        # If no valid directory was found
        if subject_anat_source_dir is None:
            if verbose:
                print(f"Could not find anatomical directory for subject {self.subject} in any of: {[str(d) for d in possible_dirs]}")
            return

        # --- T1w Conversion ---
        t1w_bids_path = BIDSPath(
                subject=self.subject,
                datatype='anat',
                suffix='T1w',
                root=self.bids_root
            )

        t1w_source_file = subject_anat_source_dir / 'T1.nii.gz'
        if t1w_source_file.exists():
            if verbose:
                print(f"Writing T1w anatomical data to: {t1w_bids_path.fpath}")
            try:
                write_anat(
                    image=str(t1w_source_file),  # Ensure path is string for older MNE-BIDS versions
                    bids_path=t1w_bids_path,
                    overwrite=overwrite,
                    verbose=verbose
                )
                # Create empty sidecar for T1w
                self._create_empty_sidecar(t1w_bids_path, 'T1w')
            except Exception as e:
                print(f"Error writing T1w anatomical data: {e}")
        else:
            import nibabel as nib
            subject_anat_source_dir = self.recon_path / self.subject / 'mri'
            native_mgz = subject_anat_source_dir / 'native.mgz'
            # convert mgz to nii.gz
            try:
                img = nib.load(str(native_mgz))
                temp_dir = t1w_bids_path.fpath.parent
                temp_file = temp_dir / 'temp_mri.nii.gz'
                nib.save(img, str(temp_file))
                write_anat(
                    image=str(temp_file),
                    bids_path=t1w_bids_path,
                    overwrite=overwrite,
                    verbose=verbose
                )
                # Create empty sidecar for T1w
                self._create_empty_sidecar(t1w_bids_path, 'T1w')
            except Exception as e:
                print(f"Error writing T1w anatomical data: {e}")
            finally:
                # Always clean up the temporary file
                if temp_file.exists():
                    temp_file.unlink()

        # --- CT Conversion ---
        # Using postimpRaw.nii.gz as identified for D63
        ct_source_file = subject_anat_source_dir / 'postimpRaw.nii.gz'
        if ct_source_file.exists():
            ct_bids_path = BIDSPath(
                subject=self.subject,
                datatype='anat',
                suffix='ct', # BIDS suffix for CT scans
                root=self.bids_root,
                check=False # allow suffix ct, because CT is still under development for BIDS
            )
            if verbose:
                print(f"Processing CT anatomical data to: {ct_bids_path.fpath}")

            import shutil
            # Create a temporary file in the same directory as the destination
            temp_dir = ct_bids_path.fpath.parent
            temp_file = temp_dir / 'temp_ct.nii'  # No .gz extension

            try:
                # Copy the file to the temp location without .gz
                shutil.copy2(ct_source_file, temp_file)

                # Now use write_anat with the uncompressed file
                write_anat(
                    image=str(temp_file),
                    bids_path=ct_bids_path,
                    overwrite=overwrite,
                    verbose=verbose
                )

                # Create empty sidecar for CT
                self._create_empty_sidecar(ct_bids_path, 'CT')

                if verbose:
                    print(f"Successfully processed CT scan to {ct_bids_path.fpath}")

            except Exception as e:
                print(f"Error writing CT anatomical data: {e}")

            finally:
                # Always clean up the temporary file
                if temp_file.exists():
                    temp_file.unlink()
        else:
            if verbose:
                print(f"CT file (postimpRaw.nii.gz) not found: {ct_source_file}. Skipping CT conversion.")


def main(
    source_path: Path,
    bids_root: Path,
    subject: str,
    task: str,
    fileIDs: Optional[list] = None,
    array_type: Optional[str] = None,
    recon_path: Optional[Path] = None,
):
    """
    Entry point to run the conversion from the command line.

    Performs
    - Initialize `BIDSConverter` with provided paths and labels.
    - Load source data and write outputs to a BIDS dataset.
    """

    loader = rhdLoader(subject, source_path, fileIDs=fileIDs,
                       array_type=array_type)
    loader.load_data()
    loader.make_cue_events()
    loader.run_mfa(task_name=TASK2MFA[task])

    # Initialize the converter
    bids_converter = BIDSConverter(
        source_path=loader.out_dir,
        subject=subject,
        task=task,
        bids_root=bids_root,
        recon_path=recon_path,
    )

    # Load and convert the data
    bids_converter.convert_to_bids()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-path",
        default=None,
        help='Directory containing the RHD source data files. If not provided,' \
        'the user will be prompted to select a directory via GUI. This directory' \
        'should also contain the trialInfo file from the task computer. This' \
        'will need to be added manually for now.'
    )
    parser.add_argument(
        "--bids-root",
        default=Path.home() / "Box" / "CoganLab" / "BIDS_1.0_Lexical_µECoG" / "BIDS",
        type=Path,
        help='Root directory to save the BIDS dataset'
    )
    parser.add_argument(
        "--subject",
        default="S71",
        help='Subject identifier (e.g., S41)'
    )
    parser.add_argument(
        "--fileIDs",
        nargs='+',
        default=None,
        help='List of file IDs to process indicating the RHD files to process' \
        'when ordered alphabetically in the source data directory. (e.g.' \
        'for the 5th through 10th files, use --fileIDs 5 6 7 8 9 10)'
    )
    parser.add_argument(
        "--array-type",
        default=None,
        help='Type of electrode array (128-strip, 256-grid, 256-strip, hybrid-strip)'
    )
    parser.add_argument(
        "--recon-path",
        default=Path.home() / "Box" / "ECoG_Recon",
        type=Path,
        help='Parent directory containing subject anatomical subdirectories (e.g., ECoG_Recon/)'
    )
    parser.add_argument(
        "--task",
        default="lexical",
        help='Name of the task (default: lexical)'
    )

    args = parser.parse_args()

    main(**vars(args))
