"""Patient-specific phoneme decoding with pitch (Poisson-disk) subsampling.

For each subsampling iteration, a fresh Poisson-disk sample of channels
at the configured target pitch is drawn from the patient's chanmap.
The selected channels are intersected with whatever channels are
present in the loaded epochs (e.g. sig-channel selection may have
trimmed the full set), and a single cross-validated decoding pass is
run on the resulting subset.  This yields a per-iteration distribution
that captures both data-split variance and Poisson-sample variance.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
from bids.layout import BIDSLayout
from mne_bids import BIDSPath
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataset import PhonemeDatasetBIDS  # noqa: E402
from utils.decoders import (DimRedReshape, NaNImputer3D,  # noqa: E402
                            decodeResultsBIDS)
from preprocessing.utils.dataloaders import (load_chanmap,  # noqa: E402
                                              load_channels_tsv)
from subsampling.chanmap_utils import (chanmap_idxs_to_epoch_picks,  # noqa
                                       resolve_pitch_for_subject)
from subsampling.poisson_disk import pitch_subsample_channels  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def build_output_path(cfg, tw_epoch, results_root):
    """Construct a BIDSPath for saving pitch-subsampling decoding
    results."""
    description = cfg.description
    phase = ('Perception' if 'perception' in description.lower()
             else 'Production')
    band = cfg.suffix

    if 'zscore' in description.lower():
        norm_type = 'Zscore'
    elif 'meansub' in description.lower():
        norm_type = 'MeanSub'
    else:
        norm_type = 'raw'

    tw_fmt = ['pre' if tw_epoch[0] < 0 else 'post',
              'pre' if tw_epoch[1] < 0 else 'post']
    tw_str = (f"[{tw_fmt[0]}{abs(tw_epoch[0]):.2g},"
              f"{tw_fmt[1]}{abs(tw_epoch[1]):.2g}]")

    sig_str = 'sig' if 'sig' in cfg.datatype else 'all'

    datatype_str = ('decode(production)(patientSpecific)(pitchSubsample)')
    if cfg.compute_chance:
        datatype_str += '(chance)'

    phoneme_suffix = ('pAll' if cfg.phoneme_idx == -1
                      else f'p{cfg.phoneme_idx}')

    pitch_str = f"{cfg.pitch_mm:.3f}".rstrip('0').rstrip('.').replace('.', 'p')

    outpath = BIDSPath(
        root=results_root,
        subject=cfg.patient,
        datatype=datatype_str,
        description=(f"{band}({phase})({norm_type})"
                     f"(tw{tw_str})({sig_str}Channel)"
                     f"(pitch{pitch_str}mm)"),
        suffix=phoneme_suffix,
        extension='.h5',
        check=False,
    )
    return outpath


def _make_cv(X, y, n_folds):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True)
    try:
        list(cv.split(X, y))
        return cv
    except ValueError:
        return KFold(n_splits=n_folds, shuffle=True)


def _decode_iter(model, X, y, cv, compute_chance=False):
    if compute_chance:
        y_pred = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = np.random.permutation(y[train_idx])
            model.fit(X_train, y_train)
            y_pred[test_idx] = model.predict(X_test)
    else:
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    return balanced_accuracy_score(y, y_pred), y_pred


@hydra.main(version_base=None, config_path='config',
            config_name='decode_phonemes_pitch')
def main(cfg: DictConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration:\n{OmegaConf.to_yaml(cfg)}')

    bids_root = Path(cfg.bids_root).expanduser()
    layout_root = Path(cfg.layout_root).expanduser() if cfg.layout_root \
        else bids_root

    tw = (OmegaConf.to_container(cfg.time_window)
          if cfg.time_window is not None else [None, None])

    logger.info(f'Loading data for subject {cfg.patient}...')
    dataset = PhonemeDatasetBIDS(
        bids_root, cfg.patient, cfg.phoneme_idx, cfg.n_phons, tw,
        description=cfg.description, suffix=cfg.suffix,
        extension=cfg.extension, task=cfg.task, datatype=cfg.datatype)
    X, y = dataset.get_data()
    logger.info(f'Loaded data: X={X.shape}, y={y.shape}, '
                f'tw={dataset.twEpoch}')

    # Channel map + full channel list come from the *raw* BIDS layout
    # (channels.tsv) so we can do pitch subsampling on the full grid
    # even when sig-channel selection has trimmed the epochs.
    raw_layout = BIDSLayout(root=layout_root, derivatives=False)
    chanmap = load_chanmap(raw_layout, cfg.patient)
    full_ch_names = load_channels_tsv(raw_layout, cfg.patient)['name'].tolist()
    logger.info(f'Channel map shape: {chanmap.shape}; '
                f'full channels: {len(full_ch_names)}')

    array_type, base_pitch_mm = resolve_pitch_for_subject(cfg.patient)
    target_pitch_mm = float(cfg.pitch_mm)
    logger.info(f'Subject {cfg.patient}: array_type={array_type}, '
                f'base_pitch={base_pitch_mm:.3f}mm, '
                f'target_pitch={target_pitch_mm:.3f}mm')

    imputer = NaNImputer3D()
    pca = DimRedReshape(PCA, n_components=cfg.pca_variance)
    clf = SVC(kernel=cfg.svm_kernel, class_weight='balanced')
    model = make_pipeline(imputer, pca, clf)

    rng = np.random.default_rng(int(cfg.seed) if cfg.seed is not None
                                else None)

    n_iter = int(cfg.n_subsample_iters)
    scores = np.zeros(n_iter)
    y_preds = np.zeros((n_iter, len(y)))
    n_picked_per_iter = np.zeros(n_iter, dtype=int)

    epoch_ch_names = list(dataset.ch_names)
    for j in tqdm(range(n_iter), desc='Pitch subsample iters'):
        ch_idxs = pitch_subsample_channels(
            chanmap, base_pitch_mm, target_pitch_mm, rng=rng)
        picks = chanmap_idxs_to_epoch_picks(ch_idxs, full_ch_names,
                                            epoch_ch_names)
        if picks.size == 0:
            logger.warning(f'Iter {j+1}: no channels survived pitch '
                           'subsampling intersected with loaded epochs; '
                           'skipping iter.')
            scores[j] = np.nan
            continue
        X_sub = X[:, picks, :]
        n_picked_per_iter[j] = picks.size

        cv = _make_cv(X_sub, y, cfg.n_folds)
        score, y_pred = _decode_iter(model, X_sub, y, cv,
                                     compute_chance=cfg.compute_chance)
        scores[j] = score
        y_preds[j] = y_pred
        logger.info(f'Iter {j+1}/{n_iter}: picks={picks.size} '
                    f'bal_acc={score:.4f}')

    logger.info(f'Mean bal_acc over {n_iter} iters: '
                f'{np.nanmean(scores):.4f} '
                f'(+/- {np.nanstd(scores):.4f})')
    logger.info(f'Mean picks/iter: {np.mean(n_picked_per_iter):.1f}')

    decoder = decodeResultsBIDS(
        model=model, nFolds=cfg.n_folds, nIter=n_iter,
        scores=scores, y_preds=y_preds)

    if cfg.results_root is not None:
        results_root = Path(cfg.results_root).expanduser()
    else:
        results_root = bids_root.parent / 'decoding'

    outpath = build_output_path(cfg, dataset.twEpoch, results_root)
    outpath.mkdir(exist_ok=True)
    logger.info(f'Saving results to {outpath.fpath}')
    decoder.save_results(outpath.fpath, overwrite=True)


if __name__ == '__main__':
    main()
