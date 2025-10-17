"""
Functions for loading various source data files.
"""

from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import scipy.io as sio
import numpy as np


def find_subject_files(source_dir: Union[str, Path], subject_id: str, audio_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """
    Find all relevant files for a subject by searching recursively from the source directory.

    Args:
        source_dir: Root directory to search from (e.g., '.../S1')
        subject_id: Subject ID (e.g., 'S1')

    Returns:
        Dict with keys: 'experiment_mat', 'ieeg_dat', 'trials_mat' pointing to the respective files.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    source_dir = Path(source_dir).resolve()  # Convert to absolute path
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = {}

    # 0. Find subject directory
    subject_dir = source_dir / subject_id
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    # 1. Find experiment.mat (should be in a 'mat' subdirectory)
    experiment_mats = list(subject_dir.rglob('mat/experiment.mat'))
    if not experiment_mats:
        raise FileNotFoundError(f"No experiment.mat found in {subject_dir}/**/mat/")
    files['experiment_mat'] = experiment_mats[0]

    # 2. Find .ieeg.dat file (should be in a subdirectory like '080318/001/')
    ieeg_files = list(subject_dir.rglob(f'**/*{subject_id}*.ieeg.dat'))
    if not ieeg_files:
        raise FileNotFoundError(f"No .ieeg.dat file found for subject {subject_id} in {subject_dir}")
    files['ieeg_dat'] = ieeg_files[0]

    # 3. Find Trials.mat (in a 'mat' subdirectory under the same date as the .ieeg.dat file)
    # Extract date from the ieeg file path (assuming format like '.../080318/001/...')
    date_part = ieeg_files[0].parent.parent.name  # Gets '080318' from the path
    trials_path = subject_dir / date_part / 'mat' / 'Trial_bids.mat'

    if not trials_path.exists():
        # Fallback: search for any Trials.mat in any mat subdirectory
        trials_files = list(subject_dir.rglob('**/mat/Trial_bids.mat'))
        if not trials_files:
            raise FileNotFoundError(f"No Trial_bids.mat found in {subject_dir}/**/mat/")
        files['trials_mat'] = trials_files[0]
    else:
        files['trials_mat'] = trials_path

    # 4. Find audio files (should be in a subdirectory like '080318/001/')
    if audio_dir is not None:
        audio_dir = Path(audio_dir)/subject_id
        audio_files = list(audio_dir.rglob(f'**/*.wav'))
        if not audio_files:
            raise FileNotFoundError(f"No audio files found for subject {subject_id} in {audio_dir}")
        files['audio_wav'] = audio_files[0]

    return files

def load_experiment_info(experiment_mat_path: Path) -> Dict[str, Any]:
    """
    Load and parse experiment.mat file to extract recording information.

    Args:
        experiment_mat_path: Path to the experiment.mat file.

    Returns:
        A dictionary containing the extracted experiment information including:
        - sampling_rate: The sampling frequency in Hz.
        - channels: List of channel names or indices.
        - channel_map: The electrode grid layout (if available).
        - Any other relevant metadata from the experiment structure.

    Raises:
        FileNotFoundError: If the experiment.mat file doesn't exist.
        KeyError: If required fields are missing from the .mat file.
    """
    if not experiment_mat_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_mat_path}")

    # Load the .mat file
    mat_data = sio.loadmat(
        str(experiment_mat_path),
        squeeze_me=True,
        struct_as_record=True,
        simplify_cells=True
    )

    # Extract the experiment structure
    experiment = mat_data.get('experiment')
    if experiment is None:
        raise KeyError("No 'experiment' structure found in the .mat file")

    # Extract sampling rate (assuming it's in experiment.processing.ieeg.sample_rate)
    try:
        sampling_rate = float(experiment['processing']['ieeg']['sample_rate'])
    except (KeyError, TypeError) as e:
        raise KeyError("Could not extract sampling rate from experiment.mat") from e

    # Extract channel information
    try:
        # Assuming channels are stored as an array in experiment.channels
        channels = list(map(str, experiment['channels']))
    except (KeyError, TypeError) as e:
        raise KeyError("Could not extract channel information from experiment.mat") from e

    # Extract channel map if available
    channel_map = experiment.get('recording', {}).get('channel_map', None)

    # Prepare the output dictionary
    experiment_info = {
        'sampling_rate': sampling_rate,
        'channels': channels,
        'channel_map': channel_map,
        'raw_experiment': experiment  # Include the full experiment structure for reference
    }

    return experiment_info


def load_ieeg_data(ieeg_file: Path, experiment_info: Dict[str, Any]) -> 'mne.io.Raw':
    """
    Load iEEG data from a .dat file using open_dat_file from ieeg.io.

    Args:
        ieeg_file: Path to the .dat file containing iEEG data
        experiment_info: Dictionary containing experiment information, including
                        'channels' and 'sampling_rate'

    Returns:
        mne.io.Raw: An MNE Raw object containing the iEEG data

    Raises:
        FileNotFoundError: If the ieeg_file does not exist
        ImportError: If ieeg package is not available
        ValueError: If required keys are missing from experiment_info
    """
    # Check if file exists
    ieeg_file = Path(ieeg_file)
    if not ieeg_file.exists():
        raise FileNotFoundError(f"iEEG data file not found: {ieeg_file}")

    # Check for required experiment info
    required_keys = ['channels', 'sampling_rate']
    for key in required_keys:
        if key not in experiment_info:
            raise ValueError(f"Missing required key in experiment_info: {key}")

    # Import open_dat_file (this will raise ImportError if ieeg is not available)
    try:
        from ieeg.io import open_dat_file
    except ImportError as e:
        raise ImportError(
            "Failed to import open_dat_file from ieeg.io. "
            "Make sure the ieeg package is installed."
        ) from e

    # Get channel names and sampling rate
    channels = [str(ch) for ch in experiment_info['channels']]  # Ensure channels are strings
    sfreq = float(experiment_info['sampling_rate'])  # Ensure sampling rate is float

    # Load the iEEG data
    raw = open_dat_file(
        str(ieeg_file),  # Convert Path to string for compatibility
        channels=channels,
        sfreq=sfreq,
        types='ecog'  # Assuming ECoG data type
    )

    return raw


def load_audio_data(audio_file: Path) -> np.ndarray:
    """
    Load audio data from a .wav file.

    Args:
        audio_file: Path to the .wav file containing audio data

    Returns:
        np.ndarray: An array containing the audio data
    """
    import librosa
    return librosa.load(str(audio_file))

def load_event_data(trials_mat_path: Path) -> List[Dict[str, Any]]:
    """
    Load and parse event data from a _bids.mat file.

    This function loads the MATLAB file and returns the events as a list of dictionaries,
    preserving the original field names and structure. Any standardization of field names
    should be handled by the caller based on the specific experiment configuration.

    Args:
        trials_mat_path: Path to the _bids.mat file

    Returns:
        List[Dict[str, Any]]: A list of event dictionaries, where each dictionary
                            contains the original fields from the .mat file, plus a
                            'source_file' field indicating the source file.

    Raises:
        FileNotFoundError: If the trials_mat_path does not exist
        ValueError: If there's an error loading or parsing the .mat file
    """
    # Check if file exists
    trials_mat_path = Path(trials_mat_path)
    if not trials_mat_path.exists():
        raise FileNotFoundError(f"Trial_bids.mat file not found: {trials_mat_path}")

    # Load the .mat file
    try:
        mat_data = sio.loadmat(
            str(trials_mat_path),
            squeeze_me=True,
            struct_as_record=True,
            simplify_cells=True
        )
    except Exception as e:
        raise ValueError(f"Failed to load .mat file {trials_mat_path}: {str(e)}")

    # Check if 'Trials' key exists in the loaded data
    if 'Trials' not in mat_data:
        # If no 'Trials' field, try to use the root level as events
        # (some .mat files might have the data at the root level)
        trials = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        if not trials:
            raise ValueError(f"No 'Trials' field or valid event data found in {trials_mat_path}")
    else:
        trials = mat_data['Trials']

    events = []

    # Handle different possible structures of the Trials data
    if isinstance(trials, dict):
        # If it's a dictionary, convert to list of dictionaries
        # This handles the case where each field is a separate array
        if not trials:  # Empty dictionary
            return []

        # Get the length from the first array-like value
        first_val = next(iter(trials.values()))
        if hasattr(first_val, '__len__') and not isinstance(first_val, str):
            trial_count = len(first_val)
        else:
            trial_count = 1

        for i in range(trial_count):
            event = {}
            for key, values in trials.items():
                if not key.startswith('__'):  # Skip metadata fields
                    if hasattr(values, '__len__') and not isinstance(values, str) and len(values) > i:
                        event[key] = values[i]
                    else:
                        event[key] = values
            events.append(event)

    elif isinstance(trials, (list, np.ndarray)) and len(trials) > 0:
        # If it's already a list/array of dictionaries/objects
        for trial in trials:
            if hasattr(trial, '_fieldnames'):  # If it's a structured array
                event = {field: getattr(trial, field) for field in trial._fieldnames}
            elif isinstance(trial, dict):  # If it's already a dictionary
                event = trial
            else:
                # If it's a simple value, create a dict with a default key
                event = {'value': trial}
            events.append(event)
    else:
        # If it's a single value, create a single event
        events = [{'value': trials}]

    # Ensure each event has a source file reference
    for event in events:
        event['source_file'] = str(trials_mat_path.name)

    return events
