"""Generate spectrograms for each channel and display them on a channel map.

Saves intermediate spectrogram data and final channel map figure to derivative
folder.
"""
from omegaconf.base import DictKeyType
import sys
import logging

import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from mne_bids import BIDSPath
from pathlib import Path
from ieeg.timefreq.multitaper import spectrogram
from ieeg.timefreq.utils import crop_pad
from mne.time_frequency import AverageTFRArray
from ieeg.viz.ensemble import plot_specChanMap
from bids.layout import BIDSLayout
from ieeg.navigate import trial_ieeg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.utils.config import normalize_cfg_items
from preprocessing.utils.preprocessing import (get_events, get_good_trials,
                                               preprocess_raw)
from preprocessing.utils.dataloaders import load_chanmap, load_raw

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def get_spectrogram(epoch, freq, epochs_base, pad="500ms"):
    spectra = spectrogram(epoch.copy(), freq, epochs_base, verbose=10,
                          n_cycles=freq / 2, pad=pad, n_jobs=-1,
                          picks=epoch.ch_names)
    return spectra


def get_precomputed_spectra(patient, bids_layout, task_periods=None):
    """Check for and load pre-computed spectrograms for each task period.

    Returns a dict mapping task period names to spectra objects, or None.
    """
    logger.info("Checking for pre-computed spectrograms...")

    patient_spec_dir = (Path(bids_layout.root) / 'derivatives' / 'spec'
                        / f"sub-{patient}" / "multitaperSpec")

    if not patient_spec_dir.exists():
        logger.info(f"No spectrogram directory found at {patient_spec_dir}")
        return None

    patient_spec_files = list(
        patient_spec_dir.glob(f"sub-{patient}_task-*_desc-*_spec.h5"))

    if not patient_spec_files:
        logger.info(f"No pre-computed spectrograms found for patient {patient}")
        return None

    found_task_periods = {}
    for spec_file in patient_spec_files:
        parts = spec_file.stem.split('_')
        task_period = None
        for part in parts:
            if part.startswith('desc-'):
                task_period = part.replace('desc-', '')
                break
        if task_period:
            found_task_periods[task_period] = spec_file
            logger.info(f"Found pre-computed spectrogram for task period: "
                        f"{task_period}")

    if not found_task_periods:
        logger.info("No valid task period spectrograms found")
        return None

    if task_periods is not None:
        requested_names = [tp.name for tp in task_periods]
        filtered_spectra = {}
        missing_periods = []

        for tp_name in requested_names:
            if tp_name in found_task_periods:
                try:
                    spectra = mne.time_frequency.read_tfrs(
                        found_task_periods[tp_name])
                    filtered_spectra[tp_name] = spectra
                    logger.info(f"Loaded pre-computed spectrogram for task "
                                f"period: {tp_name}")
                except Exception as e:
                    logger.warning(f"Failed to load spectrogram for "
                                   f"{tp_name}: {e}")
                    missing_periods.append(tp_name)
            else:
                missing_periods.append(tp_name)

        if missing_periods:
            logger.warning(f"Missing pre-computed spectrograms for task "
                           f"periods: {missing_periods}")
            logger.info("Will compute spectrograms for missing task periods")

        return filtered_spectra if filtered_spectra else None

    return found_task_periods


@hydra.main(version_base=None, config_path="config",
            config_name="generate_spec_chanMap")
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
    
    # Check for pre-computed spectrograms
    precomputed_spectra = {}
    if not cfg.recompute:
        logger.info("Searching for pre-computed spectrograms...")
        precomputed_spectra = get_precomputed_spectra(
            cfg.patient, bids_layout, task_periods)
        if not precomputed_spectra:
            logger.info("No pre-computed spectrograms found. "
                        "Will compute all spectrograms...")
            cfg.recompute = True
        else:
            logger.info(f"Found {len(precomputed_spectra)} pre-computed "
                        "spectrograms")

    if cfg.recompute or len(precomputed_spectra) < len(task_periods):
        # Full preprocessing: load -> outliers -> notch -> re-reference
        raw = preprocess_raw(bids_layout, cfg.patient, cfg.reference,
                             OmegaConf.to_container(cfg.load_kwargs))

        stim_events, _ = get_events(raw, 'stimulus')
        resp_events, _ = get_events(raw, 'response')

        def get_tp_events(task_period):
            if task_period == 'stimulus':
                return stim_events
            elif task_period == 'response':
                return resp_events
            else:
                logger.error(f'Unknown task period: {task_period}')
                raise ValueError(f"Unknown task period: {task_period}")

        epochs_base = trial_ieeg(
            raw, event=stim_events, times=[-1, 0.5], preload=True,
        )

        epochs_trial = trial_ieeg(
            raw, event=stim_events, times=[-1, 3], preload=True,
        )

        ts = cfg.trial_selection
        good_trials = get_good_trials(epochs_trial._data,
                                      threshold=ts.threshold,
                                      method=ts.method,
                                      chan_thresh=ts.chan_thresh)

        epochs_base = epochs_base[good_trials]

        spec_outpath = BIDSPath(
            root=Path(bids_layout.root) / 'derivatives' / 'spec',
            subject=cfg.patient,
            task=cfg.task,
            datatype='multitaperSpec',
            check=False
        )
        spec_outpath.mkdir(exist_ok=True)

        all_spectra = {}

        for tp_cfg in task_periods:
            logger.info(f'##### Processing task period: {tp_cfg.name} #####')

            if precomputed_spectra and tp_cfg.name in precomputed_spectra:
                logger.info(f'Using pre-computed spectrogram for {tp_cfg.name}')
                spectra = precomputed_spectra[tp_cfg.name]
            else:
                logger.info(f'Computing spectrogram for {tp_cfg.name}')
                tp_events = get_tp_events(tp_cfg.align_to)

                epoch = trial_ieeg(
                    raw, event=tp_events, times=tp_cfg.full_times,
                    preload=True,
                )
                epoch = epoch[good_trials]

                freqs = instantiate(cfg.frequencies)

                logger.info(f'Calculating spectrograms for task period '
                            f'{tp_cfg.name}...')
                spectra = get_spectrogram(epoch, freqs, epochs_base,
                                          pad="500ms")
                logger.info('... calculated spectrograms')

                spec_path = spec_outpath.copy().update(
                    suffix="spec", extension=".h5",
                    description=tp_cfg.name, check=False
                )
                spectra.save(spec_path, overwrite=True)
                logger.info(f'Saved spectrogram information to {spec_path}')

            all_spectra[tp_cfg.name] = spectra

    else:
        logger.info("All spectrograms were pre-computed.")
        logger.info("Using pre-computed data for plotting.")
        all_spectra = precomputed_spectra

    chan_map = load_chanmap(bids_layout, cfg.patient)

    fig_outpath = BIDSPath(
        root=Path(bids_layout.root) / 'derivatives' / 'figs',
        subject=cfg.patient,
        task=cfg.task,
        datatype='specChanMap',
        check=False
    )
    fig_outpath.mkdir(exist_ok=True)

    sns.set_theme(font='Arial', font_scale=1.2, style='white')
    sns.set_style(rc={
        'axes.linewidth': 0.5,
        'xtick.bottom': True,
        'ytick.left': True,
    })
    plt.rcParams['svg.fonttype'] = 'none'

    n_rows, n_cols = chan_map.shape
    base_fig_width = 8
    base_fig_height = 6
    width_per_chan = 0.8
    height_per_chan = 0.6
    fig_width = base_fig_width + width_per_chan * n_cols
    fig_height = base_fig_height + height_per_chan * n_rows

    for tp_cfg in task_periods:
        if tp_cfg.name in all_spectra:
            spectra_to_plot = all_spectra[tp_cfg.name]

            fig = plot_specChanMap(
                spectra_to_plot,
                chan_map,
                figsize=(fig_width, fig_height),
                vmin=-4,
                vmax=4,
                cmap=sns.color_palette("vlag", as_cmap=True)
            )
            fig.savefig(
                fig_outpath.copy().update(
                    suffix="specChanMap", extension=".svg",
                    description=tp_cfg.name, check=False
                ),
                format='svg',
                facecolor='white'
            )
            plt.close(fig)
            logger.info(f'Saved channel map figure for task period '
                        f'{tp_cfg.name}')


if __name__ == "__main__":
    main()
