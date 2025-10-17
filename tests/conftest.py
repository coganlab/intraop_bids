"""
Pytest configuration and fixtures for testing the ECoG BIDS conversion pipeline.

"""

import os
import shutil
from pathlib import Path
import numpy as np
import scipy.io as sio
import pytest

# Add the parent directory to the path so we can import our package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / 'data'
TEST_SUBJECT = 'test01'
TEST_SESSION = '01'

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return TEST_DATA_DIR

@pytest.fixture(scope="session")
def test_subject_dir(test_data_dir):
    """Return the path to the test subject directory."""
    return test_data_dir / f"sub-{TEST_SUBJECT}"

@pytest.fixture(scope="session")
def create_test_experiment_mat(test_data_dir):
    """Create a test experiment.mat file and return its path."""
    # Create subject directory if it doesn't exist
    subj_dir = test_data_dir / f"sub-{TEST_SUBJECT}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple experiment structure
    experiment = {
        'processing': {
            'ieeg': {
                'sample_rate': 1000.0,  # 1 kHz sampling rate
            }
        },
        'channels': np.array([1, 2, 3, 4, 5]),  # 5 channels
        'recording': {
            'channel_map': np.array([
                [1, 2, 3],
                [4, 5, np.nan]  # 2x3 grid with one empty position
            ])
        }
    }

    # Save to file
    mat_path = subj_dir / 'experiment.mat'
    sio.savemat(
        str(mat_path),
        {'experiment': experiment},
        appendmat=False
    )

    return mat_path

@pytest.fixture(scope="session")
def create_test_ieeg_dat(test_data_dir):
    """Create a test .ieeg.dat file and return its path."""
    subj_dir = test_data_dir / f"sub-{TEST_SUBJECT}"

    # Create a simple iEEG data file (5 channels, 1000 samples)
    n_channels = 5
    n_samples = 1000
    data = np.random.randn(n_channels, n_samples).astype(np.float32)

    # Save to file
    dat_path = subj_dir / f"{TEST_SUBJECT}_task-test_run-01_ieeg.dat"
    data.tofile(dat_path)

    return dat_path

@pytest.fixture(scope="session")
def create_test_trials_mat(test_data_dir):
    """Create a test Trials.mat file and return its path."""
    subj_dir = test_data_dir / f"sub-{TEST_SUBJECT}"

    # Create a simple trials structure
    n_trials = 10
    trial_dtype = [
        ('Trial', 'O'),
        ('Auditory', 'f8'),
        ('ResponseOnset', 'f8'),
        ('Correct', 'f8')
    ]

    trials = np.zeros(n_trials, dtype=trial_dtype)
    for i in range(n_trials):
        trials[i] = (
            f"trial_{i+1:02d}",  # Trial name
            i * 2.0,              # Auditory onset (s)
            i * 2.0 + 1.5,         # Response onset (s)
            1.0                    # Correct (1) or not (0)
        )

    # Save to file
    trials_path = subj_dir / 'Trials.mat'
    sio.savemat(
        str(trials_path),
        {'Trials': trials},
        appendmat=False
    )

    return trials_path

@pytest.fixture(scope="function")
def temp_bids_dir(tmp_path_factory):
    """Create a temporary BIDS directory for testing."""
    bids_dir = tmp_path_factory.mktemp("bids_output")
    return bids_dir

@pytest.fixture(scope="session")
def test_subject_files(create_test_experiment_mat, create_test_ieeg_dat, create_test_trials_mat):
    """Return a dictionary of test file paths."""
    return {
        'experiment_mat': create_test_experiment_mat,
        'ieeg_dat': create_test_ieeg_dat,
        'trials_mat': create_test_trials_mat
    }
