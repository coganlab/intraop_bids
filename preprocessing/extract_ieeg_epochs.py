import mne
import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from mne_bids import BIDSPath
from pathlib import Path
from ieeg.calc.scaling import rescale
from ieeg.io import raw_from_layout
from bids.layout import BIDSLayout
from ieeg.navigate import trial_ieeg
import logging
import sys
from utils.stats import (find_channel_outliers, remove_bad_trials,
                         get_significant_channels)
from utils.referencing import (set_white_matter_reference,
                               set_laplacian_reference)

# Simple logging: everything INFO and above to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_raw(bids_layout, subject, **kwargs):
    # search query
    raw = raw_from_layout(
        bids_layout,
        subject=subject,
        extension='.edf',
        # desc=None,
        **kwargs
    )

    # check the raw data was found with query
    if not raw:
        logger.error(f'No raw data found for subject: {subject}')
        raise ValueError(f'No raw data found for subject: {subject}')

    raw.load_data()
    raw._data *= 1e6  # convert to uV

    # remove Trigger channel we have any
    raw.drop_channels(['Trigger']) if 'Trigger' in raw.ch_names else None

    return raw


def get_events(raw, filter_str):
    logger.info('Getting events from annotations with filter string: '
                f'{filter_str}')
    events, event_id = mne.events_from_annotations(raw)
    sel_events = [e for e in event_id.keys() if filter_str in e]
    logger.info(f'Found events: {sel_events}')
    return sel_events, event_id


def set_bad_channels(raw, bad_channels):
    logger.info(f'Adding bad channels to raw data: {bad_channels}')
    bad_channels_tot = list(set(raw.info['bads'] + bad_channels))
    # update bads in raw objects with new outliers
    raw.info['bads'] = bad_channels_tot
    logger.info(f'New bad channels: {raw.info['bads']}')


def remove_powerline_noise(raw):
    # Notch filter power line noise
    logger.info('Removing powerline noise via notch filter at powerline '
                'frequency and harmonics')
    powerline_freq = raw.info.get("line_freq")
    if powerline_freq is None:
        logger.warning('Line frequency not found in raw.info, using 60 Hz')
        powerline_freq = 60
    else:
        logger.info(f'Using detected powerline frequency" {powerline_freq} Hz')
    freqs = [powerline_freq * m for m in range(1, 4)]
    raw.notch_filter(freqs=freqs, notch_widths=2, n_jobs=-1)


def set_reference(raw, ref='CAR'):
    logger.info(f'Setting re-reference scheme: {ref}')
    if ref.lower() in ['car', 'average', 'common_average', 'common average']:
        # CAR will automatically ignore channels marked in bads
        ch_type = raw.get_channel_types(only_data_chs=True)[0]
        raw = raw.set_eeg_reference(ref_channels="average", ch_type=ch_type)
    elif ref.lower() in ['wm', 'white_matter', 'white matter']:
        raw = set_white_matter_reference(raw)
    elif ref.lower() in ['lap', 'laplacian']:
        raw = set_laplacian_reference(raw)
    else:
        logger.error(f'Unknown reference: {ref}')
        raise ValueError(f"Unknown reference: {ref}")
    return raw


def get_good_trials(data, threshold=10, method=1, chan_thresh=0.8):
    # Gets good trials for each channel
    _, good_trials = remove_bad_trials(data, threshold=threshold,
                                       method=method)

    # Keep trials that are good on 80% of channels
    nChans = data.shape[1]
    assert good_trials.shape[0] == nChans
    good_trials_common = np.sum(good_trials, axis=0) >= (chan_thresh * nChans)

    return good_trials_common


def calc_sig_channels(data, baseline, tw_interest=[-0.25, 0.25],
                      tw_baseline=[-0.5, 0]):
    logger.info('Calculating significant channels')
    sig_channels = get_significant_channels(data, baseline,
                                            tw_interest=tw_interest,
                                            tw_baseline=tw_baseline)
    logger.info(f'Found {len(sig_channels)} significant channels')
    sig_channels = [str(int(ch)) for ch in sig_channels]
    return sig_channels


def snake2camel(snake_str):
    str_parts = snake_str.split('_')
    result = str_parts[0]
    if len(str_parts) > 1:
        result += ''.join([part.capitalize() for part in str_parts[1:]])
    return result


def load_config(item, config_type):
    """Load config from YAML if item is a string."""
    if isinstance(item, str):
        config_dir = Path(__file__).resolve().parent / 'config' / config_type
        config_file = config_dir / f'{item}.yaml'
        return OmegaConf.load(config_file)
    return item


def normalize_cfg_items(item, config_type):
    """Normalize Hydra overrides into a list of config objects."""
    if isinstance(item, str):
        item = item.strip()
        if item.startswith('[') and item.endswith(']'):
            return normalize_cfg_items(OmegaConf.create(item), config_type)
        return [load_config(item, config_type)]

    if isinstance(item, DictConfig):
        return [item]

    if OmegaConf.is_list(item) or isinstance(item, (list, tuple)):
        return [load_config(x, config_type) for x in item]

    if OmegaConf.is_dict(item):
        return [item]

    return [item]


@hydra.main(version_base=None, config_path="config",
            config_name="extract_ieeg_epochs")
def main(cfg: DictConfig) -> None:
    # Check for missing keys in config
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration: {OmegaConf.to_yaml(cfg)}')

    # Convert to path and expand user on root
    cfg.bids_root = Path(cfg.bids_root).expanduser()

    # Create BIDSLayout
    bids_layout = BIDSLayout(
        root=cfg.bids_root,
    )

    # Load data
    raw = load_raw(bids_layout, cfg.patient, **cfg.load_kwargs)

    # Identify channel outliers
    bad_channels_detrend = find_channel_outliers(raw)
    logger.info(f"IEEG outlier channels (detrend): {bad_channels_detrend}")
    set_bad_channels(raw, bad_channels_detrend)

    # Track channel type info
    ch_type = raw.get_channel_types(only_data_chs=True)[0]
    logger.info(f'Available channel types: {ch_type}')

    # Remove 60 Hz noise
    remove_powerline_noise(raw)

    # Re-reference data
    raw = set_reference(raw, cfg.reference)

    # Extract events aligned to stimulus and patient response onset
    stim_events, _ = get_events(raw, 'stimulus')
    resp_events, _ = get_events(raw, 'response')

    # Create function to allow for ease of event choice below
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
        raw,
        event=stim_events,
        times=[-1, 0.5],
        preload=True,
    )

    # Extract general trial epochs for shared trial outlier removal
    epochs_trial = trial_ieeg(
        raw,
        event=stim_events,
        times=[-1, 3],
        preload=True,
    )

    # Bad trial removal - Gets good trials for all task periods
    good_trials = get_good_trials(epochs_trial._data)

    # Load task period configs from strings / Hydra lists
    task_periods = normalize_cfg_items(cfg.task_periods, 'cfg_task_periods')

    # Load feature configs from strings / Hydra lists
    features = normalize_cfg_items(cfg.features, 'cfg_features')

    # Create output directory
    outpath = BIDSPath(
        root=Path(bids_layout.root) / 'derivatives' / f'epoch({cfg.reference})',
        subject=cfg.patient,
        task='lexical',
        datatype='epoch(raw)',
        check=False
    )
    outpath.mkdir(exist_ok=True)

    epochs_base.save(
        outpath.copy().update(
            suffix="raw",
            extension=".fif",
            description="baseline",
            check=False
        ),
        overwrite=True
    )

    for tp_cfg in task_periods:
        logger.info(f'##### Processing task period: {tp_cfg.name} #####')

        tp_events = get_tp_events(tp_cfg.align_to)

        epoch = trial_ieeg(
            raw,
            event=tp_events,
            times=tp_cfg.full_times,
            preload=True,
        )
        epoch = epoch[good_trials]
        epoch.save(
            outpath.copy().update(
                datatype='epoch(raw)',
                suffix="raw",
                extension=".fif",
                description=tp_cfg.name,
                check=False
            ),
            overwrite=True
        )

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
                sig_channels = calc_sig_channels(feat_data, base_data,
                                                 tw_interest=tp_cfg.sig_times)
                # remove bad channels
                sig_channels_clean = [ch for ch in sig_channels if ch not in
                                      raw.info['bads']]
                logger.info(f'Found {len(sig_channels_clean)} significant '
                            'channels after removing bad channels')
                if len(sig_channels_clean) <= 0:
                    logger.warning('No signifcant channels found for '
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

            # Resample to lower frequency for output
            logger.info(f'Resampling {feat_cfg.name} to {feat_cfg.out_fs} Hz')
            base_data = base_data.resample(sfreq=feat_cfg.out_fs, n_jobs=-1)
            feat_z = feat_z.resample(sfreq=feat_cfg.out_fs, n_jobs=-1)
            feat_ms = feat_ms.resample(sfreq=feat_cfg.out_fs, n_jobs=-1)

            if cfg.sig_channels:
                # pick only significant channels
                base_data.pick(sig_channels_clean)
                feat_data.pick(sig_channels_clean)
                feat_z.pick(sig_channels_clean)
                feat_ms.pick(sig_channels_clean)

            # Save band-specific epochs with run information
            datatype = 'epoch(band)(power)(sig)' if cfg.sig_channels else \
                       'epoch(band)(power)'
            outpath = outpath.update(
                datatype=datatype,
                suffix=snake2camel(feat_cfg.name),
                extension=".fif",
                check=False
            )
            outpath.mkdir(exist_ok=True)

            # Data saving for current task period and feature
            logger.info(f'Saving {tp_cfg.name} {feat_cfg.name} data to '
                        f'{outpath}...')
            base_data.save(outpath.copy().update(description="baseline"),
                           overwrite=True)
            feat_data.save(outpath.copy().update(description=tp_cfg.name),
                           overwrite=True)
            feat_z.save(outpath.copy().update(
                                        description=(tp_cfg.name + 'Zscore')),
                        overwrite=True)
            feat_ms.save(outpath.copy().update(
                                        description=(tp_cfg.name + 'MeanSub')),
                         overwrite=True)
            logger.info('...saving successful')


if __name__ == "__main__":
    main()
