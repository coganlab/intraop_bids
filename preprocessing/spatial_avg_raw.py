"""Spatially average raw iEEG data to simulate larger-contact recordings.

This script reads the *unprocessed* raw BIDS recording for a subject,
averages micro-contact signals into virtual macro-contact groups for
each configured ``contact_size``, and writes each averaged recording to
its own BIDS derivative at::

    <bids_root>/derivatives/spatialAvg/sub-XX/ieeg/
        sub-XX_task-Y_desc-contact{N}_ieeg.edf
        ...channels.tsv, ...events.tsv, ...ieeg.json sidecars

No preprocessing (outlier detection, notch, re-referencing) is applied
*before* averaging.  The intent is to faithfully simulate physically
larger contacts: averaging in the time domain happens first, then the
standard preprocessing pipeline (``extract_ieeg_epochs.py`` with
``spatial_avg.enabled=true``) is run on the averaged derivative as if
it were a fresh, lower-density recording.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import mne
import numpy as np
import pandas as pd
from bids.layout import BIDSLayout
from mne_bids import BIDSPath, write_raw_bids
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataloaders import load_chanmap, load_raw  # noqa: E402

from subsampling.spatial_avg import (  # noqa: E402
    average_group_indices, averaged_channel_names, spatial_average_data,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _build_averaged_raw(raw: mne.io.BaseRaw,
                        groups,
                        ch_names_new) -> mne.io.BaseRaw:
    """Construct a new MNE ``RawArray`` containing averaged channels."""
    data = raw.get_data()
    # Reference dataloader converts to uV (*1e6) on load; convert back to V
    # for the new Raw so subsequent write_raw_bids stores correct units.
    data = data / 1e6
    avg_data = spatial_average_data(data, groups)

    info = mne.create_info(
        ch_names=list(ch_names_new),
        sfreq=raw.info['sfreq'],
        ch_types='ecog',
    )
    if raw.info.get('line_freq') is not None:
        info['line_freq'] = raw.info['line_freq']

    new_raw = mne.io.RawArray(avg_data, info, first_samp=raw.first_samp,
                              verbose='ERROR')
    new_raw.set_meas_date(raw.info.get('meas_date'))
    if raw.annotations is not None and len(raw.annotations) > 0:
        new_raw.set_annotations(raw.annotations)
    return new_raw


def _write_dataset_description(deriv_root: Path):
    """Ensure derivatives/spatialAvg/dataset_description.json exists."""
    deriv_root.mkdir(parents=True, exist_ok=True)
    desc_path = deriv_root / 'dataset_description.json'
    if not desc_path.exists():
        desc = {
            "Name": "Spatial average raw derivative",
            "BIDSVersion": "1.6.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{
                "Name": "intraop_bids.preprocessing.spatial_avg_raw",
                "Description": ("Time-domain spatial averaging of "
                                "micro-contacts to simulate larger-contact "
                                "recordings; no preprocessing applied "
                                "before averaging."),
            }],
        }
        with open(desc_path, 'w') as f:
            json.dump(desc, f, indent=2)


def _update_channels_tsv(bids_path: BIDSPath, electrode_type: str,
                         contact_size: int):
    """Add ``electrode_type`` and ``contact_size`` columns to channels.tsv."""
    ch_path = bids_path.copy().update(suffix='channels', extension='.tsv')
    if not ch_path.fpath.exists():
        return
    df = pd.read_csv(ch_path.fpath, sep='\t')
    df['electrode_type'] = electrode_type
    df['contact_size'] = contact_size
    df.to_csv(ch_path.fpath, sep='\t', index=False)


def _write_chanmap_tsv(out_dir: Path, subject: str, task: str,
                       contact_size: int, reduced_rc, centroids,
                       ch_names_new):
    """Persist the reduced channel map alongside the derivative.

    Columns: ``channel_number``, ``name``, ``row``, ``col``,
    ``orig_centroid_row``, ``orig_centroid_col``, ``electrode_type``.
    """
    rows = []
    for i, (rc, cent, name) in enumerate(
            zip(reduced_rc, centroids, ch_names_new)):
        rows.append({
            'channel_number': i,
            'name': name,
            'row': int(rc[0]),
            'col': int(rc[1]),
            'orig_centroid_row': int(cent[0]),
            'orig_centroid_col': int(cent[1]),
            'electrode_type': 'averaged',
            'contact_size': contact_size,
        })
    tsv = pd.DataFrame(rows)
    out_path = out_dir / (f'sub-{subject}_task-{task}'
                          f'_desc-contact{contact_size}_channelMap.tsv')
    tsv.to_csv(out_path, sep='\t', index=False)
    logger.info(f'Wrote reduced channel map: {out_path}')


def write_spatial_avg_derivative(raw: mne.io.BaseRaw,
                                 chanmap: np.ndarray,
                                 bids_root: Path,
                                 subject: str,
                                 task: str,
                                 contact_size: int,
                                 overwrite: bool = True):
    """Generate and write one spatial-average derivative for ``contact_size``.

    Returns the ``BIDSPath`` of the written iEEG file.
    """
    logger.info(f'Computing averaging groups for contact_size={contact_size}')
    groups, reduced_rc, centroids = average_group_indices(chanmap,
                                                          contact_size)
    if len(groups) == 0:
        logger.warning(f'No valid averaging groups for contact_size='
                       f'{contact_size}; skipping.')
        return None
    ch_names_new = averaged_channel_names(reduced_rc)
    logger.info(f'Reduced channel count: {len(groups)} '
                f'(from {chanmap.size} original grid cells)')

    new_raw = _build_averaged_raw(raw, groups, ch_names_new)

    deriv_root = bids_root / 'derivatives' / 'spatialAvg'
    _write_dataset_description(deriv_root)

    bids_path = BIDSPath(
        root=deriv_root,
        subject=subject,
        task=task,
        datatype='ieeg',
        suffix='ieeg',
        description=f'contact{contact_size}',
        check=False,
    )
    bids_path.mkdir(exist_ok=True)

    write_raw_bids(
        raw=new_raw,
        bids_path=bids_path,
        overwrite=overwrite,
        allow_preload=True,
        format='EDF',
        verbose=False,
    )
    logger.info(f'Wrote averaged raw: {bids_path.fpath}')

    _update_channels_tsv(bids_path, electrode_type='averaged',
                         contact_size=contact_size)
    _write_chanmap_tsv(
        out_dir=bids_path.fpath.parent, subject=subject, task=task,
        contact_size=contact_size, reduced_rc=reduced_rc,
        centroids=centroids, ch_names_new=ch_names_new,
    )

    return bids_path


@hydra.main(version_base=None, config_path='config',
            config_name='spatial_avg_raw')
def main(cfg: DictConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration:\n{OmegaConf.to_yaml(cfg)}')

    bids_root = Path(cfg.bids_root).expanduser()
    layout = BIDSLayout(root=bids_root, derivatives=False)

    logger.info(f'Loading raw (unprocessed) for {cfg.patient}...')
    load_kwargs = OmegaConf.to_container(cfg.load_kwargs)
    raw = load_raw(layout, cfg.patient, **load_kwargs)
    logger.info(f'Raw: {len(raw.ch_names)} channels, '
                f'sfreq={raw.info["sfreq"]} Hz')

    chanmap = load_chanmap(layout, cfg.patient)
    logger.info(f'Channel map shape: {chanmap.shape}')

    contact_sizes = list(OmegaConf.to_container(cfg.contact_sizes))
    overwrite = bool(cfg.overwrite)

    for contact_size in contact_sizes:
        write_spatial_avg_derivative(
            raw=raw, chanmap=chanmap, bids_root=bids_root,
            subject=cfg.patient, task=cfg.task,
            contact_size=int(contact_size), overwrite=overwrite,
        )


if __name__ == '__main__':
    main()
