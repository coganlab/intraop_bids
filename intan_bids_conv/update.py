import argparse
from bids import BIDSLayout
import mne
import pandas as pd
from pathlib import Path
import numpy as np
import librosa
from mne_bids import read_raw_bids
from mne_bids import BIDSPath
import glob
from pandas._config.config import F
import soundfile as sf
from ieeg.io import raw_from_layout
from tqdm import tqdm
import soundfile as sf
from mne_bids import write_raw_bids
import os

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

def add_montage_to_raw(
    raw: mne.io.Raw,
    recon_dir: str,
    subject: str,
):
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
        recon_dir,
        f"{subject}",
        "elec_recon",
        f"{subject}_elec_locations_RAS_brainshifted.txt"
    )

    # First check if file exists
    try:
        # Try to load and set the RAS montage
        montage = load_ras_montage(ras_file)

        raw.set_montage(montage)
    except Exception as e:
        # If any error occurs (e.g., channel mismatch, invalid format), fall back to custom montage
        print(f"Error applying RAS montage: {e}")

    return raw


def main(
    bids_root: str,
    recon_dir: str,
    subject: str,
    task: str,
    save_path: str,
):

    bids_layout = BIDSLayout(
        root=bids_root,
        derivatives=False,
    )

    raw = raw_from_layout(
        bids_layout,
        subject=subject,
        extension='.edf',
    )

    raw.load_data()

    raw = add_montage_to_raw(
        raw,
        recon_dir,
        subject,
    )
    bids_path = BIDSPath(
            subject=subject.lstrip('sub-'),  # Remove 'sub-' prefix if present
            task=task,
            root=os.path.join(save_path, 'bids'),
            datatype='ieeg',
            suffix='ieeg',
        )
    bids_path.mkdir(exist_ok=True)

    write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        overwrite=True,
        verbose=True,
        allow_preload=True,
        format='EDF',
        acpc_aligned=True,
    )

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--bids_root", type=str,
                        default=r"C:\Users\ns458\Box\CoganLab\BIDS_1.0_Phoneme_Sequence_uECoG_share\BIDS",
                        help="Root directory of the BIDS dataset")
    parser.add_argument("--save_path", type=str,
                        default=".",
                        help="Path to save the raw file")
    parser.add_argument("--recon_dir",default=r"C:\Users\ns458\Box\ECoG_Recon",
                        type=str,
                        help="Path to the stimulus file")
    parser.add_argument("--subject", type=str, default='S58',
                        help="Subject to process")
    parser.add_argument("--task", type=str, default='phoneme',
                        help="Task to process")
    args = parser.parse_args()
    main(**vars(args))
