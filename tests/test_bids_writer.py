"""
Tests for the bids_writer module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import mne
from mne import create_info
from mne.io import RawArray

from ecog_bids_pipeline.bids_writer import BIDSConverter

# Skip these tests if mne_bids is not available
pytest.importorskip("mne_bids")

# Fixtures
@pytest.fixture
def mock_raw():
    """Create a mock Raw object for testing."""
    n_channels = 5
    n_samples = 1000
    sfreq = 1000.0  # 1 kHz
    data = np.random.randn(n_channels, n_samples)
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    ch_types = ['ecog'] * n_channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return RawArray(data, info)

@pytest.fixture
def mock_experiment_info():
    """Create mock experiment info."""
    return {
        'sampling_rate': 1000.0,
        'recording_date': '2025-01-01',
        'channels': [f'EEG{i:03d}' for i in range(5)]
    }

@pytest.fixture
def mock_events():
    """Create mock events."""
    return np.array([
        [100, 0, 1],
        [200, 0, 2],
        [300, 0, 1]
    ])

@pytest.fixture
def mock_find_files():
    """Mock the find_subject_files function."""
    with patch('ecog_bids_pipeline.bids_writer.find_subject_files') as mock:
        mock.return_value = {
            'experiment_mat': Path('/mock/path/experiment.mat'),
            'ieeg_dat': Path('/mock/path/subject1.ieeg.dat'),
            'trials_mat': Path('/mock/path/Trials.mat')
        }
        yield mock

@pytest.fixture
def mock_load_functions(mock_raw, mock_experiment_info, mock_events):
    """Mock the loader functions."""
    with patch.multiple('ecog_bids_pipeline.bids_writer',
                       load_experiment_info=MagicMock(return_value=mock_experiment_info),
                       load_ieeg_data=MagicMock(return_value=mock_raw),
                       load_event_data=MagicMock(return_value=mock_events)):
        yield

def test_bids_converter_initialization():
    """Test BIDSConverter initialization."""
    converter = BIDSConverter(
        source_dir="/mock/source",
        subject_id="S1",
        task_name="testtask",
        bids_root="/mock/bids"
    )
    
    assert converter.source_dir == Path("/mock/source")
    assert converter.subject_id == "S1"
    assert converter.task_name == "testtask"
    assert converter.bids_root == Path("/mock/bids")
    assert converter.raw is None
    assert converter.events is None
    assert converter.experiment_info is None
    assert converter.bids_path is None

def test_bids_converter_load_data(mock_find_files, mock_load_functions):
    """Test loading data with BIDSConverter."""
    converter = BIDSConverter(
        source_dir="/mock/source",
        subject_id="S1",
        task_name="testtask"
    )
    
    # This will use the mocked functions
    converter.load_data()
    
    # Verify the loader functions were called with correct paths
    mock_find_files.assert_called_once_with(Path("/mock/source"), "S1")
    
    # Verify data was loaded
    assert converter.raw is not None
    assert converter.events is not None
    assert converter.experiment_info is not None
    assert 'sampling_rate' in converter.experiment_info
    assert 'recording_date' in converter.experiment_info

def test_generate_dataset_description(temp_bids_dir):
    """Test generating a dataset_description.json file."""
    converter = BIDSConverter(
        source_dir="/mock/source",
        subject_id="S1",
        task_name="testtask",
        bids_root=temp_bids_dir
    )
    
    output_path = converter.generate_dataset_description(
        name="Test Dataset",
        authors=["Test Author"],
        description="Test dataset for BIDS conversion"
    )
    
    # Check that the file was created
    assert output_path.exists()
    assert output_path.name == "dataset_description.json"
    
    # Verify content
    with open(output_path, 'r') as f:
        content = f.read()
        assert '"Name": "Test Dataset"' in content
        assert 'Test Author' in content
        assert '"Description": "Test dataset for BIDS conversion"' in content

def test_bids_directory_structure(temp_bids_dir):
    """Test BIDS directory structure without session."""
    converter = BIDSConverter(
        source_dir="/mock/source",
        subject_id="test01",
        task_name="testtask",
        bids_root=temp_bids_dir
    )
    
    # Create the directory structure
    subject_dir = converter.create_bids_directory()
    
    # Verify the directory structure
    assert subject_dir.exists()
    assert subject_dir.name == "sub-test01"
    
    # Check subdirectories
    for modality in ["anat", "ieeg"]:
        assert (subject_dir / modality).exists()
    
    # Verify the directory is returned correctly
    assert subject_dir == temp_bids_dir / "sub-test01"

def test_convert_to_bids(temp_bids_dir, mock_find_files, mock_load_functions, mock_raw):
    """Test the complete BIDS conversion workflow."""
    # Initialize converter
    converter = BIDSConverter(
        source_dir="/mock/source",
        subject_id="S1",
        task_name="testtask",
        bids_root=temp_bids_dir
    )
    
    # Load the data (uses mocked loaders)
    converter.load_data()
    
    # Convert to BIDS
    bids_path = converter.convert_to_bids(overwrite=True)
    
    # Check that the main file was created
    assert bids_path.fpath.exists()
    
    # Check that sidecar files were created
    for suffix in ['channels', 'events', 'ieeg']:
        sidecar_path = bids_path.copy().update(
            suffix=suffix,
            extension='.tsv' if suffix != 'ieeg' else '.json',
            check=False
        )
        assert sidecar_path.fpath.exists(), f"Sidecar file not found: {sidecar_path.fpath.name}"
    
    # Verify the output directory structure
    expected_path = temp_bids_dir / "sub-S1" / "ieeg" / "sub-S1_task-testtask_ieeg.edf"
    assert expected_path.exists()

def test_update_sidecar_files(temp_bids_dir, mock_find_files, mock_load_functions):
    """Test updating sidecar files with metadata."""
    converter = BIDSConverter(
        source_dir="/mock/source",
        subject_id="S1",
        task_name="testtask",
        bids_root=temp_bids_dir
    )
    
    # Load data and convert to BIDS
    converter.load_data()
    bids_path = converter.convert_to_bids()
    
    # Update sidecar files
    converter.update_sidecar_files()
    
    # Check that the sidecar file was updated with task info
    sidecar_path = bids_path.copy().update(suffix='ieeg', extension='.json')
    assert sidecar_path.fpath.exists()
    
    # Verify content
    with open(sidecar_path.fpath, 'r') as f:
        content = f.read()
        assert '"TaskName": "testtask"' in content
        assert '"SamplingFrequency": 1000.0' in content

def test_error_handling():
    """Test error handling in BIDSConverter methods."""
    # Test missing required parameters
    with pytest.raises(TypeError):
        BIDSConverter()  # Missing required parameters
    
    # Test with minimal required parameters
    converter = BIDSConverter(
        source_dir="/mock/source",
        subject_id="S1",
        task_name="testtask"
    )
    
    # Test calling convert_to_bids without loading data first
    with pytest.raises(ValueError, match="No data loaded"):
        converter.convert_to_bids()
    
    # Test update_sidecar_files without BIDS path
    with pytest.raises(ValueError, match="BIDS path not set"):
        converter.update_sidecar_files(None)  # None for bids_path
