"""
Configuration and constants for the ECoG BIDS conversion pipeline.
"""

from pathlib import Path

# BIDS version to use
BIDS_VERSION = "1.8.0"

# Default task name if not specified
DEFAULT_TASK_NAME = "lexical"

# Power line frequency (Hz) - adjust based on your recording location
POWER_LINE_FREQUENCY = 60  # 50 Hz for Europe/Asia, 60 Hz for North America

# Default coordinate frame for electrode positions
DEFAULT_COORD_FRAME = "Other"


DEFAULT_CHANNEL_MAP = 'ecog'

# Default metadata for dataset_description.json
DEFAULT_DATASET_DESCRIPTION = {
    "Name": "ECoG Phoneme/Word Processing Dataset",
    "BIDSVersion": BIDS_VERSION,
    "Authors": ["Researcher Name"],
    "HowToAcknowledge": "Please cite this dataset if you use it in your research.",
    "Funding": ["Funding Source"],
    "ReferencesAndLinks": [""],
    "DatasetDOI": "",
}

# Default hardware filter settings for ieeg.json
DEFAULT_HARDWARE_FILTERS = {
    "HighpassFilter": {"CutoffFrequency": 0.15},
    "LowpassFilter": {"CutoffFrequency": 500},
}


# Default event processing configuration
DEFAULT_EVENT_PROCESSING = {
    "sampling_rate_source": "raw_info",  # Defaulting to raw.info['sfreq']
    "bids_events_columns": {
        "onset": {
            "source_field": "Auditory",  # Column name in your Trials.mat file
            "unit": "seconds"  # Unit of the data in 'source_field'. Options: "seconds" or "samples".
                               # This will be converted to samples for MNE and written in seconds for BIDS.
        },
        "duration": {
            "value": 0  # Fixed value for the duration column in BIDS events.tsv (in seconds).
                       # MNE point events inherently have 0 duration.
        },
        "trial_type": {
            "source_field": "Stimulus" # This will be the 'trial_type' column in BIDS events.tsv
        }
    }
}
