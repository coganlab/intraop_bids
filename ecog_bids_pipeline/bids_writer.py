"""
Classes for writing data to BIDS format using mne-bids.
"""
from pathlib import Path
from typing import Optional, Dict, Union, List, Any
from mne_bids import BIDSPath, write_raw_bids, write_anat
import numpy as np
from ecog_bids_pipeline.loaders import (
    find_subject_files,
    load_ieeg_data,
    load_experiment_info,
    load_event_data,
    load_audio_data,
)
from ecog_bids_pipeline.config import (
    DEFAULT_EVENT_PROCESSING,
    DEFAULT_TASK_NAME,
)

class BIDSConverter:
    """
    Main class for converting ECoG data to BIDS format.

    This class handles the creation of BIDS directory structure, conversion of
    raw data, and generation of BIDS-compliant metadata files.
    """

    def __init__(
        self,
        subject_id: str,
        task_name: str,
        source_dir: Union[str, Path],
        bids_root: Optional[Union[str, Path]] = None,
        audio_dir: Optional[Union[str, Path]] = None,
        anat_dir: Optional[Union[str, Path]] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the BIDS converter with source data information.

        Args:
            source_dir: Directory containing the source data files
            subject_id: Subject identifier (e.g., 'S1')
            task_name: Name of the task (e.g., 'lexical')
            bids_root: Root directory for the BIDS dataset
            session_id: Optional session identifier (e.g., 'pre', 'post', '01')
            audio_dir: Optional directory containing audio files
            anat_dir: Optional directory containing anatomical files
        """
        self.subject_id = subject_id
        self.task_name = task_name
        self.session_id = session_id
        self.source_dir = Path(source_dir)
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.anat_dir = Path(anat_dir) if anat_dir else None
        self.bids_root = Path(bids_root) if bids_root else None

        # Will be populated when data is loaded
        self.experiment_info = None
        self.raw = None
        self.events = None
        self.bids_path = None

    def load_data(self):
        """
        Load all required data using the loader functions.

        Returns:
            self: Returns the instance for method chaining

        Raises:
            FileNotFoundError: If any required files are missing
            ValueError: If there's an error loading the data
        """
        # Find all required files
        files = find_subject_files(self.source_dir, self.subject_id)

        # Load experiment metadata
        self.experiment_info = load_experiment_info(files['experiment_mat'])

        # Load iEEG data
        self.raw = load_ieeg_data(files['ieeg_dat'], self.experiment_info)

        # Load events
        self.events = load_event_data(files['trials_mat'])

        if self.audio_dir is not None:
            wav_dir = find_subject_files(self.source_dir, self.subject_id, self.audio_dir)['audio_wav']
            self.audio_files, self.audio_fs = load_audio_data(wav_dir)

        return self

    def update_raw_object(self):
        """Update the raw object with processed events."""
        self.events, self.event_id = self.handling_event()

        # make custom montage if no montage is found in the raw object
        if self.raw.info.get_montage() is None:
            self.raw = self.add_montage_to_raw()

        return self.raw

    def add_montage_to_raw(self):
        """Add a montage to the raw object."""
        # load RAS file from anat directory
        # ras_file = self.anat_dir / f"{self.subject_id}/elec_recon/{self.subject_id}_elec_locations_RAS_brainshifted.txt"
        ras_file = self.anat_dir / f"{self.subject_id}/eleecon/{self.subject_id}_elec_locations_RAS_brainshifted.txt"

        # First check if file exists
        if not ras_file.exists():
            print(f"RAS file not found: {ras_file}, using custom montage as placeholder")
            montage = self.make_custom_montage(self.experiment_info)
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
            montage = self.make_custom_montage(self.experiment_info)
            self.raw.set_montage(montage)

        return self.raw

    @staticmethod
    def load_ras_montage(ras_file):
        """
        Load a montage from a RAS file and include hemisphere information.

        Parameters
        ----------
        ras_file : pathlib.Path
            Path to the RAS file containing electrode coordinates and hemisphere info

        Returns
        -------
        mne.channels.DigMontage
            MNE montage object with electrode positions and hemisphere information
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

        # Create channel names by combining prefix and number
        ch_names = df['number'].astype(str)

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
    def make_custom_montage(experiment_info):
        """Create a custom montage for the iEEG data."""
        from mne.channels import Layout, make_dig_montage
        channel_map = experiment_info['channel_map']
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
        """Process raw event data into MNE-compatible events and event_id mapping.

        Returns:
            tuple: (events_array, event_id_dict) where:
                - events_array: numpy array of shape (n_events, 3)
                - event_id_dict: mapping from event names to integer IDs
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

    def create_bids_directory(self) -> Path:
        """
        Create the BIDS directory structure.

        Returns:
            Path: Path to the created subject or session directory.

        Raises:
            ValueError: If bids_root or subject_id is not set
        """
        if self.bids_root is None:
            raise ValueError("bids_root must be set to create BIDS directory structure")
        if not hasattr(self, 'subject_id'):
            raise ValueError("subject_id must be set")

        # Create main BIDS directories
        self.bids_root.mkdir(parents=True, exist_ok=True)

        # Create subject directory (sub-<label>)
        subject_dir = self.bids_root / f"sub-{self.subject_id}"
        subject_dir.mkdir(exist_ok=True)

        # Anatomical data goes in subject root (session-less)
        (subject_dir / "anat").mkdir(exist_ok=True)

        # For iEEG data, create session directory if session_id is provided
        if self.session_id is not None:
            session_dir = subject_dir / f"ses-{self.session_id}"
            session_dir.mkdir(exist_ok=True)
            (session_dir / "ieeg").mkdir(exist_ok=True, parents=True)
            return session_dir
        else:
            # For session-less data, create ieeg directory in subject root
            (subject_dir / "ieeg").mkdir(exist_ok=True)
            return subject_dir


    def save_to_derivative(
        self,
        data: Any,
        folder: str,
        filename: str,
        file_type: str = 'wav',
        description: str = 'raw',
        **kwargs
    ) -> Optional[Path]:

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
            run: Optional[Union[str, int]] = None,
            overwrite: bool = True,
            verbose: bool = True
    ) -> BIDSPath:
        """
        Convert data to BIDS format using instance attributes.

        Args:
            output_dir: Optional root directory for BIDS dataset. Uses self.bids_root if None.
            run: Optional run number.
            overwrite: Whether to overwrite existing files.
            verbose: Whether to print progress messages.

        Returns:
            BIDSPath to the created file.

        Raises:
            ValueError: If data is not loaded or required attributes are missing
        """
        if self.raw is None:
            raise ValueError("No data loaded. Call load_data() first.")

        bids_root = Path(output_dir) if output_dir else self.bids_root
        if bids_root is None:
            raise ValueError("output_dir must be provided or bids_root must be set")

        # Create BIDS directory structure
        self.create_bids_directory()

        # Create BIDS path with session if provided
        bids_path = BIDSPath(
            subject=self.subject_id.lstrip('sub-'),  # Remove 'sub-' prefix if present
            session=self.session_id,
            task=self.task_name,
            acquisition='01',
            root=str(bids_root),
            run=str(run) if run is not None else '01',
            datatype='ieeg',
            suffix='ieeg',
        )

        # Create output directory if it doesn't exist
        bids_path.mkdir(exist_ok=True)

        # update raw
        raw = self.update_raw_object()

        # Write the data to BIDS format
        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            events=self.events,
            event_id=self.event_id,
            overwrite=overwrite,
            verbose=verbose,
            allow_preload=True,
            format='EDF',
            acpc_aligned=True
        )

        # Store the bids_path for later use
        self.bids_path = bids_path

        if hasattr(self, 'audio_files') and self.audio_files is not None:
            self.save_to_derivative(
                data=self.audio_files,
                folder='audio',
                filename='microphone',
                file_type='wav',
                overwrite=overwrite,
                verbose=verbose
            )

        self._convert_anat_to_bids(overwrite=overwrite, verbose=verbose)

        return self.bids_path

    def _create_empty_sidecar(self, bids_path: BIDSPath, modality: str):
        """Create an empty JSON sidecar file for anatomical data.

        Args:
            bids_path: The BIDSPath object for the anatomical file
            modality: The modality ('T1w' or 'CT')
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
        Convert anatomical data (T1w and CT) to BIDS format.

        Args:
            overwrite: Whether to overwrite existing anatomical files.
            verbose: Whether to print verbose output.
        """
        possible_dirs = [
            self.anat_dir / self.subject_id / 'elec_recon',
            self.anat_dir / self.subject_id / 'mri',
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
                print(f"Could not find anatomical directory for subject {self.subject_id} in any of: {[str(d) for d in possible_dirs]}")
            return

        # --- T1w Conversion ---
        t1w_bids_path = BIDSPath(
                subject=self.subject_id,
                session=self.session_id,
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
            subject_anat_source_dir = self.anat_dir / self.subject_id / 'mri'
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
                subject=self.subject_id,
                session=self.session_id, # CT is often co-registered to T1w of a specific session
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


def parse_arguments():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert ECoG data to BIDS format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--source-dir',
        default=Path("/Users/shinanlin/Library/CloudStorage/Box-Box/CoganLab/Data/Micro/Processed Data"),
        type=Path,
        help='Directory containing the source data files'
    )
    required.add_argument(
        '--bids-root',
        default=Path("./bids/lexical"),
        type=Path,
        help='Root directory for the BIDS dataset'
    )
    required.add_argument(
        '--subject',
        default="S52",
        help='Subject identifier (e.g., S33)'
    )

    parser.add_argument(
        '--audio-dir',
        default=Path("/Users/shinanlin/Library/CloudStorage/Box-Box/CoganLab/Data/Micro/microphone"),
        type=Path,
        help='Directory containing audio files (optional)'
    )
    parser.add_argument(
        '--anat-dir',
        default=Path("/Users/shinanlin/Library/CloudStorage/Box-Box/ECoG_Recon"),
        type=Path,
        help='Parent directory containing subject anatomical subdirectories (e.g., ECoG_Recon/)'
    )
    parser.add_argument(
        '--task',
        default=DEFAULT_TASK_NAME,
        help=f'Name of the task (default: {DEFAULT_TASK_NAME})'
    )
    parser.add_argument(
        '--session',
        default=None,
        help='Session identifier (e.g., 01, pre, post)'
    )
    parser.add_argument(
        '--run',
        type=int,
        default=None,
        help='Run number (for multiple runs of the same task)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=True,
        help='Overwrite existing BIDS files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Print verbose output'
    )

    return parser.parse_args()


def main():
    """Run the BIDS conversion with command line arguments."""
    args = parse_arguments()

    if args.verbose:
        print(f"Starting BIDS conversion for subject {args.subject}")
        print(f"Source directory: {args.source_dir}")
        print(f"BIDS root: {args.bids_root}")
        if args.audio_dir:
            print(f"Audio directory: {args.audio_dir}")
        if args.anat_dir: # Add this for verbose output
            print(f"Anatomical directory: {args.anat_dir}")

    try:
        # Initialize the converter
        bids_converter = BIDSConverter(
            source_dir=args.source_dir,
            subject_id=args.subject,
            task_name=args.task,
            session_id=args.session,
            bids_root=args.bids_root,
            audio_dir=args.audio_dir,
            anat_dir=args.anat_dir # Add anat_dir here
        )

        # Load and convert the data
        bids_converter.load_data()
        bids_path = bids_converter.convert_to_bids(
            run=args.run,
            overwrite=args.overwrite,
            verbose=args.verbose
        )

        if args.verbose:
            print(f"\nBIDS conversion complete!")
            print(f"Output files written to: {bids_path.directory}")

    except Exception as e:
        print(f"Error during BIDS conversion: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    main()