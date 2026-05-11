import sys
import logging

import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from mne_bids import BIDSPath
from pathlib import Path
from ieeg.calc.scaling import rescale
from bids.layout import BIDSLayout
from ieeg.navigate import trial_ieeg

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import snake2camel
from utils.config import normalize_cfg_items
from utils.preprocessing import (get_events, get_bad_trial_mask,
                                 preprocess_raw,
                                 preprocess_raw_data,
                                 preprocess_derivative_raw)
from utils.dataloaders import load_derivative_raw
from utils.phoneme import remap_phoneme_events
from utils.stats import get_significant_channels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _apply_nan_mask(data, good_mask):
    """Set bad trial-channel data to NaN in-place.

    Args:
        data: np.ndarray of shape (n_trials, n_channels, n_times).
        good_mask: np.ndarray of shape (n_channels, n_trials).
            True indicates a good (artifact-free) trial for that channel.
    """
    for ch_idx in range(data.shape[1]):
        bad_trials = ~good_mask[ch_idx]
        data[bad_trials, ch_idx, :] = np.nan


def calc_sig_channels(data, baseline, tw_interest=[-0.25, 0.25],
                      tw_baseline=[-0.5, 0]):
    logger.info('Calculating significant channels')
    sig_channels = get_significant_channels(data, baseline,
                                            tw_interest=tw_interest,
                                            tw_baseline=tw_baseline)
    logger.info(f'Found {len(sig_channels)} significant channels')
    # sig_channels = [str(int(ch)) for ch in sig_channels]
    return sig_channels


@hydra.main(version_base=None, config_path="config",
            config_name="extract_ieeg_epochs")
def main(cfg: DictConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration: {OmegaConf.to_yaml(cfg)}')

    cfg.bids_root = Path(cfg.bids_root).expanduser()

    spatial_avg_cfg = cfg.get('spatial_avg', None)
    spatial_avg_enabled = bool(spatial_avg_cfg
                               and spatial_avg_cfg.get('enabled', False))
    if spatial_avg_enabled and spatial_avg_cfg.get('contact_size') is None:
        raise ValueError(
            'spatial_avg.contact_size must be set when '
            'spatial_avg.enabled=true')

    # Include derivatives in the layout if we need to read from any of them.
    bids_layout = BIDSLayout(
        root=cfg.bids_root,
        derivatives=bool(cfg.phoneme_level) or spatial_avg_enabled,
    )

    if spatial_avg_enabled:
        contact_size = int(spatial_avg_cfg.contact_size)
        deriv_name = spatial_avg_cfg.get('derivative_name', 'spatialAvg')
        desc = f'contact{contact_size}'
        logger.info(f'##### Spatial-average mode (contact_size='
                    f'{contact_size}) #####')
        logger.info(f'Loading averaged-but-unprocessed derivative raw '
                    f'from derivatives/{deriv_name} (desc={desc})')
        raw = load_derivative_raw(
            bids_layout, cfg.patient, derivative_name=deriv_name,
            desc=desc, **OmegaConf.to_container(cfg.load_kwargs))
        # Now apply the standard preprocessing chain to the averaged
        # recording as if it came from a real macro-contact array.
        raw = preprocess_raw_data(raw, cfg.reference)
    else:
        # Full preprocessing: load -> outliers -> notch -> re-reference
        raw = preprocess_raw(bids_layout, cfg.patient, cfg.reference,
                             OmegaConf.to_container(cfg.load_kwargs))

    # Extract events aligned to stimulus and patient response onset
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

    # Extract epochs for baseline period
    epochs_base = trial_ieeg(
        raw, event=stim_events, times=[-1, 0.5], preload=True,
    )

    # Extract general trial epochs for shared trial outlier removal
    # epochs_trial = trial_ieeg(
    #     raw, event=stim_events, times=[-1, 3], preload=True,
    # )
    epochs_trial = trial_ieeg(
        raw, event=resp_events, times=[-3, 3], preload=True,
    )

    # Per-channel bad trial detection
    ts = cfg.trial_selection
    good_mask = get_bad_trial_mask(epochs_trial._data,
                                   threshold=ts.threshold,
                                   method=ts.method)
    n_good_per_ch = np.sum(good_mask, axis=1)
    logger.info(f'Good trials per channel: min={n_good_per_ch.min()}, '
                f'max={n_good_per_ch.max()}, mean={n_good_per_ch.mean():.1f}')

    n_trials_total = len(epochs_base)

    # Load task period and feature configs
    task_periods = normalize_cfg_items(cfg.task_periods, 'cfg_task_periods')
    features = normalize_cfg_items(cfg.features, 'cfg_features')

    # Create output directory
    if spatial_avg_enabled:
        epoch_dir = (f'epoch(spatialAvgContact{int(spatial_avg_cfg.contact_size)})'
                     f'({cfg.reference})')
    elif cfg.phoneme_level:
        epoch_dir = f'epoch(phonemeLevel)({cfg.reference})'
    else:
        epoch_dir = f'epoch({cfg.reference})'
    outpath = BIDSPath(
        root=Path(bids_layout.root) / 'derivatives' / epoch_dir,
        subject=cfg.patient,
        task=cfg.task,
        datatype='epoch(raw)',
        check=False
    )
    outpath.mkdir(exist_ok=True)

    baseline_outpath = outpath.copy().update(
        suffix="raw", extension=".fif",
        description="baseline", check=False
    )
    epochs_base_save = epochs_base.copy()
    _apply_nan_mask(epochs_base_save._data, good_mask)
    epochs_base_save.save(baseline_outpath, overwrite=True)
    del epochs_base_save

    mask_save_dir = (Path(bids_layout.root) / 'derivatives' / 'good_trials'
                     / f'sub-{cfg.patient}')
    mask_save_dir.mkdir(parents=True, exist_ok=True)
    mask_save_path = (mask_save_dir
                      / f'sub-{cfg.patient}_task-{cfg.task}'
                        f'_goodTrialMask.npy')
    np.save(mask_save_path, good_mask)
    logger.info(f'Saved per-channel good trial mask to {mask_save_path}')

    good_mask_feat = good_mask
    nPhons = 1
    if cfg.phoneme_level:
        logger.info('##### Phoneme-level mode: loading derivative raw #####')
        raw_phon = load_derivative_raw(
            bids_layout, cfg.patient, cfg.derivative_name,
            **OmegaConf.to_container(cfg.load_kwargs))
        raw_phon = preprocess_derivative_raw(raw_phon, raw, cfg.reference)

        phon_resp_events, _ = get_events(raw_phon, 'response')
        phon_epochs_count = trial_ieeg(
            raw_phon, event=phon_resp_events, times=[0, 0.1], preload=True)
        nPhons = len(phon_epochs_count) // n_trials_total
        logger.info(f'Phonemes per trial: {nPhons}')

        good_mask_feat = np.repeat(good_mask, nPhons, axis=1)
        logger.info(f'Expanded good trial mask: {good_mask.shape[1]} trials '
                    f'-> {good_mask_feat.shape[1]} phoneme observations')

        raw = raw_phon
        stim_events, _ = get_events(raw, 'stimulus')
        resp_events, _ = get_events(raw, 'response')

    for tp_cfg in task_periods:
        logger.info(f'##### Processing task period: {tp_cfg.name} #####')

        tp_events = get_tp_events(tp_cfg.align_to)

        epoch = trial_ieeg(
            raw, event=tp_events, times=tp_cfg.full_times, preload=True,
        )

        epoch_save = epoch.copy()
        _apply_nan_mask(epoch_save._data, good_mask_feat)
        epoch_save.save(
            outpath.copy().update(
                datatype='epoch(raw)', suffix="raw", extension=".fif",
                description=tp_cfg.name, check=False
            ),
            overwrite=True
        )
        del epoch_save

        for feat_cfg in features:
            logger.info(f'--- Processing feature: {feat_cfg.name} ---')
            logger.info(f'Extracting {feat_cfg.name} features...')
            base_data = instantiate(feat_cfg.extractor,
                                    epochs=epochs_base.copy(),
                                    _convert_='all')
            logger.info('...extracted baseline features')

            feat_data = instantiate(feat_cfg.extractor, epochs=epoch.copy(),
                                    _convert_='all')
            logger.info('...extracted task features')

            logger.info(f'Cropping task data to {tp_cfg.feat_times[0]} to '
                        f'{tp_cfg.feat_times[1]} ms from {tp_cfg.name} anchor '
                        'onset')
            base_data = base_data.crop(tmin=-0.5, tmax=0)
            feat_data = feat_data.crop(tmin=tp_cfg.feat_times[0],
                                       tmax=tp_cfg.feat_times[1])

            if cfg.sig_channels:
                feat_data_sig = feat_data[::nPhons]
                _apply_nan_mask(feat_data_sig._data, good_mask)
                base_data_sig = base_data.copy()
                _apply_nan_mask(base_data_sig._data, good_mask)
                sig_channels = calc_sig_channels(feat_data_sig, base_data_sig,
                                                 tw_interest=tp_cfg.sig_times)
                del base_data_sig
                sig_channels_clean = [ch for ch in sig_channels if ch not in
                                      raw.info['bads']]
                logger.info(f'Found {len(sig_channels_clean)} significant '
                            'channels after removing bad channels')
                if len(sig_channels_clean) <= 0:
                    logger.warning('No significant channels found for '
                                   f'{tp_cfg.name} {feat_cfg.name} data in '
                                   f'{cfg.patient}. Skipping this feature.')
                    continue

            # Normalize by baseline
            logger.info(f'Normalizing {feat_cfg.name} features by baseline '
                        '(z-score)')
            feat_z = rescale(feat_data, base_data, mode='zscore', copy=True)
            logger.info(f'Normalizing {feat_cfg.name} features by baseline '
                        '(mean-subtraction)')
            feat_ms = rescale(feat_data, base_data, mode='mean', copy=True)

            if cfg.sig_channels:
                all_ch_names = feat_z.ch_names
                sig_ch_indices = [all_ch_names.index(ch)
                                  for ch in sig_channels_clean]

                base_data.pick(sig_channels_clean)
                feat_data.pick(sig_channels_clean)
                feat_z.pick(sig_channels_clean)
                feat_ms.pick(sig_channels_clean)

                good_mask_apply = good_mask_feat[sig_ch_indices]
                good_mask_base_apply = good_mask[sig_ch_indices]
            else:
                good_mask_apply = good_mask_feat
                good_mask_base_apply = good_mask

            # Resample to lower frequency for output
            logger.info(f'Resampling {feat_cfg.name} to {feat_cfg.out_fs} Hz')
            base_data = base_data.resample(sfreq=feat_cfg.out_fs, n_jobs=-1)
            feat_z = feat_z.resample(sfreq=feat_cfg.out_fs, n_jobs=-1)
            feat_ms = feat_ms.resample(sfreq=feat_cfg.out_fs, n_jobs=-1)

            # Apply per-channel NaN mask to mark bad trial-channel data
            _apply_nan_mask(base_data._data, good_mask_base_apply)
            _apply_nan_mask(feat_data._data, good_mask_apply)
            _apply_nan_mask(feat_z._data, good_mask_apply)
            _apply_nan_mask(feat_ms._data, good_mask_apply)

            # Save band-specific epochs
            datatype = ('epoch(band)(power)(sig)' if cfg.sig_channels else
                        'epoch(band)(power)')
            outpath = outpath.update(
                datatype=datatype,
                suffix=snake2camel(feat_cfg.name),
                extension=".fif",
                check=False
            )
            outpath.mkdir(exist_ok=True)

            logger.info(f'Saving {tp_cfg.name} {feat_cfg.name} data to '
                        f'{outpath}...')
            base_data.save(outpath.copy().update(description="baseline"),
                           overwrite=True)
            feat_data.save(outpath.copy().update(description=tp_cfg.name),
                           overwrite=True)
            feat_z.save(outpath.copy().update(
                                        description=f"{tp_cfg.name}Zscore"),
                        overwrite=True)
            feat_ms.save(outpath.copy().update(
                                        description=f"{tp_cfg.name}MeanSub"),
                         overwrite=True)
            logger.info('...saving successful')


if __name__ == "__main__":
    main()
