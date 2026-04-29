"""Generate MP4 videos of z-scored feature activity projected onto channel maps.

Searches for pre-computed feature data in derivative folders and creates videos
for each configured task period and feature combination.
"""
import sys
import logging

import mne
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from mne_bids import BIDSPath
from pathlib import Path
from bids.layout import BIDSLayout
from ieeg.viz.ensemble import plot_vals_chanMap_video

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.utils import snake2camel
from preprocessing.utils.config import normalize_cfg_items
from preprocessing.utils.dataloaders import load_chanmap

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def get_precomputed_features(patient, bids_layout, task_periods, features,
                             reference):
    """Load pre-computed z-scored feature data for each (task_period, feature).

    Returns a dict mapping ``(tp_name, feat_name)`` tuples to MNE Epochs.
    """
    logger.info("Checking for pre-computed feature data...")

    feature_base_dir = (Path(bids_layout.root) / 'derivatives'
                        / f'epoch({reference})')

    if not feature_base_dir.exists():
        logger.info(f"No feature directory found at {feature_base_dir}")
        return {}

    found_features = {}

    for tp_cfg in task_periods:
        for feat_cfg in features:
            feature_path = BIDSPath(
                root=feature_base_dir,
                subject=patient,
                task=bids_layout.get_tasks()[0] if bids_layout.get_tasks()
                        else None,
                datatype='epoch(band)(power)',
                suffix=snake2camel(feat_cfg.name),
                extension=".fif",
                description=f"{tp_cfg.name}Zscore",
                check=False
            )

            matches = feature_path.match()
            if matches:
                try:
                    epochs = mne.read_epochs(matches[0], preload=False)
                    found_features[(tp_cfg.name, feat_cfg.name)] = epochs
                    logger.info(f"Found pre-computed feature: "
                                f"{tp_cfg.name} {feat_cfg.name}")
                except Exception as e:
                    logger.warning(f"Failed to load feature "
                                    f"{tp_cfg.name} {feat_cfg.name}: {e}")
                    break
            else:
                logger.info(f"No pre-computed feature found for: "
                            f"{tp_cfg.name} {feat_cfg.name}")

    return found_features


@hydra.main(version_base=None, config_path="config",
            config_name="generate_feature_videos")
def main(cfg: DictConfig):
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration: {OmegaConf.to_yaml(cfg)}')

    cfg.bids_root = Path(cfg.bids_root).expanduser()

    bids_layout = BIDSLayout(root=cfg.bids_root)

    # Resolve config root for visualization sub-configs
    vis_config_root = Path(__file__).resolve().parent / 'config'

    task_periods = normalize_cfg_items(cfg.task_periods, 'cfg_task_periods',
                                       config_root=vis_config_root)
    features = normalize_cfg_items(cfg.features, 'cfg_features',
                                   config_root=vis_config_root)

    precomputed_features = get_precomputed_features(
        cfg.patient, bids_layout, task_periods, features, cfg.reference
    )

    if not precomputed_features:
        logger.error("No pre-computed features found. Cannot generate videos.")
        return

    try:
        chan_map = load_chanmap(bids_layout, cfg.patient)
        logger.info(f"Loaded channel map for patient {cfg.patient}")
    except Exception as e:
        logger.error(f"Failed to load channel map: {e}")
        return

    for tp_cfg in task_periods:
        for feat_cfg in features:
            feature_key = (tp_cfg.name, feat_cfg.name)

            if feature_key not in precomputed_features:
                logger.info(f"Skipping {tp_cfg.name} {feat_cfg.name} "
                            "- no pre-computed data")
                continue

            logger.info(f'##### Processing video for {tp_cfg.name} '
                        f'{feat_cfg.name} #####')

            epochs = precomputed_features[feature_key]
            data = epochs.get_data()
            mean_data = np.mean(data, axis=0)  # (channels, time_points)
            data_timeseries = mean_data.T       # (time_points, channels)

            times = epochs.times
            time_window = (times[0], times[-1])
            sampling_rate = epochs.info['sfreq']

            vid_outpath = BIDSPath(
                root=Path(bids_layout.root) / 'derivatives' / 'figs',
                subject=cfg.patient,
                task=cfg.task,
                datatype='featureVideos',
                suffix=snake2camel(feat_cfg.name),
                extension=".mp4",
                description=f"{tp_cfg.name}",
                check=False
            )
            vid_outpath.mkdir(exist_ok=True)

            title = f"{cfg.patient} - {tp_cfg.name} - {feat_cfg.name}"
            label = "Z-score"

            try:
                plot_vals_chanMap_video(
                    data_timeseries=data_timeseries,
                    chan_map=chan_map,
                    time_window=time_window,
                    title=title,
                    label=label,
                    figsize=cfg.get('figsize', (12, 8)),
                    cbar_lower=cfg.get('cbar_lower', None),
                    cbar_upper=cfg.get('cbar_upper', None),
                    cmap=cfg.get('cmap', 'vlag'),
                    fps=cfg.get('fps', 30),
                    show_video=False,
                    save_video=True,
                    output_path=str(vid_outpath),
                    sampling_rate=sampling_rate
                )
                logger.info(f'Successfully created video: {vid_outpath}')

            except Exception as e:
                logger.error(f"Failed to create video for {tp_cfg.name} "
                             f"{feat_cfg.name}: {e}")
                continue

    logger.info("Video generation complete!")


if __name__ == "__main__":
    main()
