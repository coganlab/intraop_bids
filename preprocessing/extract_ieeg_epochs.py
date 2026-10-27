import sys
import logging

import numpy as np
import pandas as pd
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
                                 )
from utils.dataloaders import load_derivative_raw, load_events_tsv
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
                      tw_baseline=[-0.5, 0], method='tpc', n_jobs=-1):
    logger.info('Calculating significant channels')
    sig_channels = get_significant_channels(data, baseline,
                                            method=method,
                                            tw_interest=tw_interest,
                                            tw_baseline=tw_baseline,
                                            n_jobs=n_jobs)
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
        # derivatives=bool(cfg.phoneme_level) or spatial_avg_enabled,
        derivatives=bool(cfg.derivative_name) or spatial_avg_enabled,
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
    delay_events, _ = get_events(raw, 'delay')
    resp_events, _ = get_events(raw, 'response')

    def get_tp_events(task_period):
        if task_period == 'stimulus':
            return stim_events
        elif task_period == 'delay':
            return delay_events
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

    # Load task period and feature configs
    task_periods = normalize_cfg_items(cfg.task_periods, 'cfg_task_periods')
    features = normalize_cfg_items(cfg.features, 'cfg_features')

    # Create output directory
    if spatial_avg_enabled:
        epoch_dir = (f'epoch(spatialAvgContact{int(spatial_avg_cfg.contact_size)})'
                     f'({cfg.reference})')
    elif cfg.derivative_name:
        epoch_dir = f'epoch({cfg.derivative_name})({cfg.reference})'
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

    if cfg.derivative_name:
        logger.info('##### Sub-event derivative mode: borrowing derivative events '
                    '#####')
        # Load the phoneme-level derivative only to extract its annotations,
        # then transplant them onto the already-preprocessed raw. This keeps
        # the current channel structure (which may be spatial-averaged) and
        # avoids the channel mismatch that would arise from swapping in the
        # phoneme-level raw, whose channels track the original recording.
        events_df_deriv = load_events_tsv(bids_layout, cfg.patient, cfg.task, derivative_name=cfg.derivative_name)
        stim_df_deriv = events_df_deriv[['stimulus' in x for x in events_df_deriv['trial_type']]]
        resp_df_deriv = events_df_deriv[['response' in x for x in events_df_deriv['trial_type']]]

        items_per_trial_stim = [sum(stim_df_deriv['trial'] == i) for i in stim_df_deriv['trial'].unique()]
        items_per_trial_resp = [sum(resp_df_deriv['trial'] == i) for i in resp_df_deriv['trial'].unique()]
        
        logger.info(f'Stimulus items per trial: {items_per_trial_stim}')
        logger.info(f'Response items per trial: {items_per_trial_resp}')

        good_mask_feat_stim = np.repeat(good_mask, items_per_trial_stim, axis=1)
        good_mask_feat_resp = np.repeat(good_mask, items_per_trial_resp, axis=1)
        logger.info(f'Expanded good trial mask: {good_mask.shape[1]} trials '
                    f'-> {good_mask_feat_stim.shape[1]} (stim) and {good_mask_feat_resp.shape[1]} (resp) items')

        raw_sub = load_derivative_raw(
            bids_layout, cfg.patient, derivative_name=cfg.derivative_name,
            desc=cfg.derivative_name, preload=False)
        raw.set_annotations(raw_sub.annotations)  # transplant derivative annotations onto preprocessed raw

        # no delay events here - only make sense on the trial level, not the sub-event level
        stim_events, _ = get_events(raw, 'stimulus')
        resp_events, _ = get_events(raw, 'response')
    else:
        items_per_trial_stim = [1] * good_mask.shape[1]
        items_per_trial_resp = [1] * good_mask.shape[1]
        good_mask_feat_stim = good_mask
        good_mask_feat_resp = good_mask

    for tp_cfg in task_periods:
        logger.info(f'##### Processing task period: {tp_cfg.name} #####')

        tp_events = get_tp_events(tp_cfg.align_to)
        good_mask_feat = (good_mask_feat_stim if tp_cfg.align_to == 'stimulus'
                          else good_mask_feat_resp)

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
                        f'{tp_cfg.feat_times[1]} s from {tp_cfg.name} anchor '
                        'onset')
            base_data = base_data.crop(tmin=-0.5, tmax=0)
            feat_data = feat_data.crop(tmin=tp_cfg.feat_times[0],
                                       tmax=tp_cfg.feat_times[1])

            if cfg.sig_channels:
                # determine entries corresponding to trial starts for each trial period
                items_per_trial = items_per_trial_stim if tp_cfg.align_to == 'stimulus' else items_per_trial_resp
                trial_starts = np.cumsum(items_per_trial) - np.array(items_per_trial)

                # use only trial-aligned data for significance
                feat_data_sig = feat_data[trial_starts]
                _apply_nan_mask(feat_data_sig._data, good_mask)
                base_data_sig = base_data.copy()
                _apply_nan_mask(base_data_sig._data, good_mask)
                sig_channels = calc_sig_channels(feat_data_sig, base_data_sig,
                                                 tw_interest=tp_cfg.sig_times,
                                                 method=cfg.sig_channels_method,
                                                 n_jobs=-1)
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
