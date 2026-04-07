# add imports
import mne
from mne_bids import get_bids_path_from_fname
import pandas as pd


def set_white_matter_reference(
    raw: mne.io.Raw
):
    """
    Set a white-matter (WM) reference using channels labeled as white matter.

    Procedure
    - Reads the BIDS `channels.tsv` sidecar corresponding to the current raw file.
    - Selects channels with `status_description == 'pure_white_matter'`.
      If none, falls back to `status_description == 'white_matter'`.
    - Applies MNE's referencing with the selected WM channels as reference.

    Parameters
    - raw (mne.io.Raw): Loaded Raw object with valid `raw.filenames[0]` path.

    Returns
    - mne.io.Raw: New Raw with WM reference applied (MNE returns a new instance).

    Notes
    - `ch_type` is inferred from the first data channel type to match the dataset (ieeg/eeg).
    - Make sure non-data channels (e.g., Trigger) are dropped prior to referencing.
    """
    # Locate the channels.tsv sidecar via BIDS path
    ref_path = get_bids_path_from_fname(raw.filenames[0])
    ref_path = ref_path.copy().update(suffix="channels", extension=".tsv").fpath

    # Read sidecar to discover WM channels
    ref_df = pd.read_csv(ref_path, sep='\t')
    # Prefer pure white matter channels; if none, use all white matter
    wm_channels = ref_df[ref_df['status_description'] == 'pure_white_matter']['name'].tolist()
    # Keep only those present in the current recording
    wm_channels = [ch for ch in wm_channels if ch in raw.ch_names]
    if not wm_channels:
        wm_channels = ref_df[ref_df['status_description'] == 'white_matter']['name'].tolist()

    # Infer channel type for referencing (e.g., 'ieeg' or 'eeg')
    ch_type = raw.get_channel_types(only_data_chs=True)[0]

    # Apply WM reference; MNE returns a new Raw instance
    raw = raw.set_eeg_reference(ref_channels=wm_channels, ch_type=ch_type)

    return raw


def set_laplacian_reference(
    raw: mne.io.Raw
):
    """
    Apply a 1D Laplacian/bipolar re-reference along each electrode shaft.

    Overview
    - Channels are grouped by shaft using a name pattern like "LA1, LA2, ..." or
      "E1, E2, ...". The regex `([a-zA-Z]+)(\d+)` extracts:
        - shaft name: alphabetic prefix (e.g., "LA", "E")
        - contact index: numeric suffix (e.g., 1, 2, ...)
    - For each shaft with >= 2 contacts:
        - First contact: V1 - V2
        - Last contact: VN - V(N-1)
        - Middle contacts: Vk - 0.5 * (V(k-1) + V(k+1)) (discrete Laplacian)

    Parameters
    - raw (mne.io.BaseRaw): MNE Raw object containing iEEG channels. Channel names
      are expected to follow the pattern described above so that grouping works.

    Returns
    - mne.io.BaseRaw: A COPY of the input with Laplacian-referenced data.

    Assumptions & Caveats
    - Channel naming must be consistent and monotonic along the shaft, e.g., LA1, LA2, ...
    - This function accesses `raw._data` directly for speed; this bypasses some of MNE's
      provenance handling. Use with care and only on loaded data (`raw.load_data()`).
    - Sorting: channel order within a shaft should be numerical (1,2,...). If channel names
      do not sort numerically by their numeric suffix, adjust the sorting logic accordingly.
    - Non-iEEG channels should be removed before calling this function.
    - Shafts with <2 contacts are skipped.

    Note on sorting implementation
    - The intent is to sort by contact number (numeric suffix). If your data structure here is a
      list of strings (channel names), consider sorting by the extracted integer index from the
      regex, e.g., `key=lambda name: int(re.match(r'([A-Za-z]+)(\d+)', name).group(2))`.
    """
    # Make a copy to avoid modifying original data
    raw = raw.copy()
    # Group channels by shafts
    shaft_groups = {}
    pattern = re.compile(r'([a-zA-Z]+)(\d+)')
    for ch_name in raw.ch_names:
        match = pattern.match(ch_name)
        if match:
            shaft_name = match.group(1) # The letter part is the shaft name
            contact_num = int(match.group(2))
            if shaft_name not in shaft_groups:
                shaft_groups[shaft_name] = []
            # Collect channel name under its shaft
            shaft_groups[shaft_name].append((ch_name, contact_num))
    for shaft_name, shaft_channels in shaft_groups.items():
        if len(shaft_channels) < 2:
            print(f"Skipping shaft '{shaft_name}' because it has fewer than 2 channels.")
            continue
        shaft_channels.sort(key=lambda x: x[1])
        shaft_channel_names = [ch[0] for ch in shaft_channels]
        
        # Store original data for this shaft to avoid in-place modification corruption
        shaft_indices = [raw.ch_names.index(ch_name) for ch_name in shaft_channel_names]
        original_data = raw._data[shaft_indices, :].copy()
        
        for i, current_ch_name in enumerate(shaft_channel_names):
            current_ch_idx = raw.ch_names.index(current_ch_name)
            if i == 0:
                # First contact: subtract next contact (bipolar)
                # V1 <- V1 - V2
                raw._data[current_ch_idx, :] = original_data[i, :] - original_data[i + 1, :]
            elif i == len(shaft_channel_names) - 1:
                # Last contact: subtract previous contact (bipolar)
                # VN <- VN - V(N-1)
                raw._data[current_ch_idx, :] = original_data[i, :] - original_data[i - 1, :]
            else:
                # Middle contacts: Laplacian (subtract average of neighbors)
                # Vk <- Vk - 0.5 * (V(k-1) + V(k+1))
                raw._data[current_ch_idx, :] = original_data[i, :] - 0.5 * (original_data[i - 1, :] + original_data[i + 1, :])
    return raw