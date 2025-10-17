"""
Tests for the loaders module.
"""

import pytest
from pathlib import Path
import numpy as np
from ecog_bids_pipeline.loaders import find_subject_files, load_experiment_info

# Real data paths for testing
REAL_SOURCE_DIR = Path("/Users/shinanlin/Library/CloudStorage/Box-Box/CoganLab/Data/Micro/Processed Data/S1")
REAL_SUBJECT_ID = "S1"  # Update if your subject ID is different

def test_find_subject_files(test_data_dir, test_subject_files):
    """Test finding subject files with test data."""
    # The test_subject_files fixture already has the paths we need
    files = test_subject_files

    print("\n=== Testing with test data directory ===")
    print(f"Test data directory: {test_data_dir}")

    # Check that all expected files are found
    assert 'experiment_mat' in files
    assert 'ieeg_dat' in files
    assert 'trials_mat' in files

    # Print the paths being tested
    print("\nFound the following files:")
    for key, path in files.items():
        print(f"- {key}: {path}")
        print(f"  Exists: {path.exists()}")

    # Verify the paths exist
    assert files['experiment_mat'].exists()
    assert files['ieeg_dat'].exists()
    assert files['trials_mat'].exists()

# Skip this test if the real data directory doesn't exist
@pytest.mark.skipif(
    not REAL_SOURCE_DIR.exists(), 
    reason=f"Real data directory not found: {REAL_SOURCE_DIR}"
)
def test_find_subject_files_real_data():
    """Test finding subject files using real data paths."""
    print(f"\n=== Testing with real data directory ===")
    print(f"Source directory: {REAL_SOURCE_DIR}")
    print(f"Subject ID: {REAL_SUBJECT_ID}")

    # Call the function with real paths
    result = find_subject_files(REAL_SOURCE_DIR, REAL_SUBJECT_ID)

    # Verify the function returns the expected dictionary keys
    expected_keys = ['experiment_mat', 'ieeg_dat', 'trials_mat']
    assert all(key in result for key in expected_keys), \
           f"Missing expected keys in result. Got: {list(result.keys())}"

    # Print detailed information about found files
    print("\nFound the following files:")
    for key, path in result.items():
        print(f"- {key}: {path}")
        print(f"  Exists: {path.exists()}")

    # Verify all files exist
    for key, path in result.items():
        assert path.exists(), f"File not found for {key}: {path}"

def test_load_experiment_info(test_subject_files):
    """Test loading experiment info from .mat file."""
    experiment_mat_path = test_subject_files['experiment_mat']
    print(f"\n=== Testing load_experiment_info ===")
    print(f"Loading experiment.mat from: {experiment_mat_path}")

    # Load the test experiment.mat file
    experiment_info = load_experiment_info(experiment_mat_path)

    # Check that required keys are present
    assert 'sampling_rate' in experiment_info
    assert 'channels' in experiment_info
    assert 'channel_map' in experiment_info
    assert 'raw_experiment' in experiment_info # Ensure the raw data is also there

    # Verify data types and values based on conftest.py:create_test_experiment_mat
    assert isinstance(experiment_info['sampling_rate'], float)
    assert experiment_info['sampling_rate'] == 1000.0

    assert isinstance(experiment_info['channels'], list)
    assert len(experiment_info['channels']) == 5
    assert experiment_info['channels'] == ['1', '2', '3', '4', '5'] # Channels are converted to str

    assert isinstance(experiment_info['channel_map'], np.ndarray) 
    # Based on conftest.py, it's a 2x3 grid. Let's check its shape.
    # Note: np.nan will make the dtype float
    assert experiment_info['channel_map'].shape == (2, 3)
    # Check a few values to be sure, including the NaN
    assert experiment_info['channel_map'][0,0] == 1
    assert experiment_info['channel_map'][1,0] == 4
    assert np.isnan(experiment_info['channel_map'][1,2])

    print("Experiment info loaded successfully and assertions passed.")

def test_inspect_trials_mat(test_subject_files):
    """Inspect the structure of Trials.mat for debugging purposes."""
    trials_mat_path = test_subject_files['trials_mat']
    print(f"\n=== Inspecting Trials.mat ===")
    print(f"Loading Trials.mat from: {trials_mat_path}")

    # Load the .mat file
    import scipy.io as sio
    mat_data = sio.loadmat(
        str(trials_mat_path),
        squeeze_me=True,
        struct_as_record=True,
        simplify_cells=True
    )

    # Print top-level keys
    print("\nTop-level keys in Trials.mat:")
    for key in mat_data.keys():
        print(f"- {key}")

    # If there's a 'Trials' key, inspect its structure
    if 'Trials' in mat_data:
        trials = mat_data['Trials']
        print("\nStructure of 'Trials' field:")

        # If it's a structured array, print its fields and data types
        if hasattr(trials, 'dtype') and trials.dtype.fields is not None:
            print("Trials is a structured array with fields:")
            for field_name, field_dtype in trials.dtype.fields.items():
                print(f"  - {field_name}: {field_dtype[0].name}")

                # For the first trial, print the value of this field
                if len(trials) > 0:
                    first_trial = trials[0]
                    field_value = getattr(first_trial, field_name, 'N/A')
                    print(f"    First trial value: {field_value}")

        # If it's a list/array of objects, print the first one's attributes
        elif isinstance(trials, (list, np.ndarray)) and len(trials) > 0:
            first_trial = trials[0]
            print("First trial object attributes:")
            for attr in dir(first_trial):
                if not attr.startswith('_'):  # Skip private attributes
                    try:
                        value = getattr(first_trial, attr)
                        print(f"  - {attr}: {type(value).__name__}")
                    except Exception as e:
                        print(f"  - {attr}: <error accessing: {str(e)}>")

    print("\nFull mat_data structure:")
    print(mat_data)

def test_load_ieeg_data(test_subject_files, test_data_dir):
    """Test loading iEEG data from .dat file using open_dat_file."""
    ieeg_dat_path = test_subject_files['ieeg_dat']
    print(f"\n=== Testing iEEG Data Loading ===")
    print(f"Loading iEEG data from: {ieeg_dat_path}")

    # Load experiment info to get channel and sampling rate info
    experiment_mat_path = test_subject_files['experiment_mat']
    experiment_info = load_experiment_info(experiment_mat_path)

    # Get channel names and sampling rate from experiment info
    channels = experiment_info['channels']
    sfreq = experiment_info['sampling_rate']

    print(f"Channels: {channels}")
    print(f"Sampling rate: {sfreq} Hz")

    # Import open_dat_file
    from ieeg.io import open_dat_file
    
    # Load the iEEG data
    print("\nLoading iEEG data with open_dat_file...")
    raw = open_dat_file(
        ieeg_dat_path,
        channels=channels,
        sfreq=sfreq,
        types='ecog'
    )

    # Print basic info about the loaded data
    print("\nSuccessfully loaded iEEG data:")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Data shape: {raw.get_data().shape}")
    print(f"Channel names: {raw.ch_names}")

    # Basic assertions
    assert len(raw.ch_names) == len(channels), "Number of channels doesn't match"
    assert raw.info['sfreq'] == sfreq, "Sampling frequency doesn't match"
    assert 'ecog' in raw.get_channel_types(), "No ECoG channels found"

    # Check data shape (channels x time points)
    data = raw.get_data()
    assert data.shape[0] == len(channels), "Number of channels in data doesn't match"
    assert data.shape[1] > 0, "No time points in the data"

    print("\niEEG data loading test passed successfully!")


def test_load_ieeg_data_function(test_subject_files):
    """Test the load_ieeg_data function with test data."""
    print("\n=== Testing load_ieeg_data function ===")

    # Get the test data paths
    ieeg_dat_path = test_subject_files['ieeg_dat']
    experiment_mat_path = test_subject_files['experiment_mat']

    print(f"Using iEEG file: {ieeg_dat_path}")
    print(f"Using experiment info from: {experiment_mat_path}")

    # Load experiment info to get channel and sampling rate info
    experiment_info = load_experiment_info(experiment_mat_path)

    # Import the function we're testing
    from ecog_bids_pipeline.loaders import load_ieeg_data

    # Test loading the iEEG data
    print("\nCalling load_ieeg_data...")
    raw = load_ieeg_data(ieeg_dat_path, experiment_info)
    
    # Print basic info about the loaded data
    print("\nSuccessfully loaded iEEG data using load_ieeg_data:")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Data shape: {raw.get_data().shape}")
    print(f"Channel names: {raw.ch_names}")
    
    # Basic assertions
    assert len(raw.ch_names) == len(experiment_info['channels']), \
        "Number of channels doesn't match"
    assert raw.info['sfreq'] == experiment_info['sampling_rate'], \
        "Sampling frequency doesn't match"
    assert 'ecog' in raw.get_channel_types(), "No ECoG channels found"
    
    # Check data shape (channels x time points)
    data = raw.get_data()
    assert data.shape[0] == len(experiment_info['channels']), \
        "Number of channels in data doesn't match"
    assert data.shape[1] > 0, "No time points in the data"
    
    print("\nload_ieeg_data function test passed successfully!")


def test_load_event_data_function(test_subject_files):
    """Test the load_event_data function with test data."""
    print("\n=== Testing load_event_data function ===")
    
    # Get the test data path
    trials_mat_path = test_subject_files['trials_mat']
    print(f"Using Trials.mat from: {trials_mat_path}")
    
    # Import the function we're testing
    from ecog_bids_pipeline.loaders import load_event_data
    
    # Test loading the event data
    print("\nCalling load_event_data...")
    events = load_event_data(trials_mat_path)
    
    # Print basic info about the loaded events
    print(f"\nSuccessfully loaded {len(events)} events")
    if events:
        print("\nFirst event:")
        for key, value in events[0].items():
            print(f"  - {key}: {value}")
    
    # Basic assertions
    assert len(events) > 0, "No events were loaded"
    
    # Check that source file is included
    assert 'source_file' in events[0], "source_file should be included in the event"
    assert events[0]['source_file'] == 'Trials.mat', "Unexpected source file name"
    
    # Check that we have the expected fields from our test data
    expected_fields = ['Trial', 'Auditory', 'ResponseOnset', 'Correct']
    for field in expected_fields:
        assert field in events[0], f"Expected field '{field}' not found in event"
    
    print("\nload_event_data function test passed successfully!")
