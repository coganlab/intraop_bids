"""Patient-specific phoneme decoding with coverage (subgrid) subsampling.

The full chanmap is tiled with all valid ``win_size`` rectangular
subgrids.  Each subgrid is treated as an independent decoding
"iteration" (channels for that placement, then one CV pass).  The
number of iterations is therefore data-driven: it equals the number of
valid placements (multiplied by ``extra_iters_per_subgrid`` if the
caller wants to also estimate split variance).
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
from subsampling.chanmap_utils import chanmap_idxs_to_epoch_picks  # noqa
from subsampling.coverage import coverage_subgrids  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def build_output_path(cfg, tw_epoch, results_root, win_size):
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

    datatype_str = 'decode(production)(patientSpecific)(coverageSubsample)'
    if cfg.compute_chance:
        datatype_str += '(chance)'

    phoneme_suffix = ('pAll' if cfg.phoneme_idx == -1
                      else f'p{cfg.phoneme_idx}')

    win_str = f'{int(win_size[0])}x{int(win_size[1])}'

    outpath = BIDSPath(
        root=results_root,
        subject=cfg.patient,
        datatype=datatype_str,
        description=(f"{band}({phase})({norm_type})"
                     f"(tw{tw_str})({sig_str}Channel)"
                     f"(win{win_str})"),
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
            config_name='decode_phonemes_coverage')
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

    raw_layout = BIDSLayout(root=layout_root, derivatives=False)
    chanmap = load_chanmap(raw_layout, cfg.patient)
    full_ch_names = load_channels_tsv(raw_layout, cfg.patient)['name'].tolist()
    logger.info(f'Channel map shape: {chanmap.shape}; '
                f'full channels: {len(full_ch_names)}')

    win_size = (int(cfg.win_size[0]), int(cfg.win_size[1]))
    subgrids = coverage_subgrids(chanmap, win_size)
    if not subgrids:
        raise ValueError(
            f'No valid {win_size} subgrids in chanmap of shape '
            f'{chanmap.shape}.')
    logger.info(f'Found {len(subgrids)} valid {win_size} subgrid placements')

    epoch_ch_names = list(dataset.ch_names)
    valid_subgrids = []
    for ch_idxs in subgrids:
        picks = chanmap_idxs_to_epoch_picks(ch_idxs, full_ch_names,
                                            epoch_ch_names)
        if picks.size > 0:
            valid_subgrids.append(picks)
    logger.info(f'{len(valid_subgrids)}/{len(subgrids)} subgrids overlap '
                f'with loaded epoch channels')

    if not valid_subgrids:
        raise ValueError('No subgrid overlaps with loaded epoch channels; '
                         'check sig-channel filter / chanmap alignment.')

    extra = int(cfg.extra_iters_per_subgrid)
    n_iter = len(valid_subgrids) * extra
    logger.info(f'Total iterations: {len(valid_subgrids)} subgrids * '
                f'{extra} repeats = {n_iter}')

    imputer = NaNImputer3D()
    pca = DimRedReshape(PCA, n_components=cfg.pca_variance)
    clf = SVC(kernel=cfg.svm_kernel, class_weight='balanced')
    model = make_pipeline(imputer, pca, clf)

    scores = np.zeros(n_iter)
    y_preds = np.zeros((n_iter, len(y)))
    n_picked_per_iter = np.zeros(n_iter, dtype=int)
    subgrid_idx_per_iter = np.zeros(n_iter, dtype=int)

    it = 0
    for sg_idx, picks in enumerate(
            tqdm(valid_subgrids, desc='Coverage subgrids')):
        for _ in range(extra):
            X_sub = X[:, picks, :]
            cv = _make_cv(X_sub, y, cfg.n_folds)
            score, y_pred = _decode_iter(model, X_sub, y, cv,
                                         compute_chance=cfg.compute_chance)
            scores[it] = score
            y_preds[it] = y_pred
            n_picked_per_iter[it] = picks.size
            subgrid_idx_per_iter[it] = sg_idx
            logger.info(f'Subgrid {sg_idx+1}/{len(valid_subgrids)} '
                        f'(iter {it+1}/{n_iter}): picks={picks.size} '
                        f'bal_acc={score:.4f}')
            it += 1

    logger.info(f'Mean bal_acc over {n_iter} iters: '
                f'{np.nanmean(scores):.4f} '
                f'(+/- {np.nanstd(scores):.4f})')

    decoder = decodeResultsBIDS(
        model=model, nFolds=cfg.n_folds, nIter=n_iter,
        scores=scores, y_preds=y_preds)

    if cfg.results_root is not None:
        results_root = Path(cfg.results_root).expanduser()
    else:
        results_root = bids_root.parent / 'decoding'

    outpath = build_output_path(cfg, dataset.twEpoch, results_root, win_size)
    outpath.mkdir(exist_ok=True)
    logger.info(f'Saving results to {outpath.fpath}')
    decoder.save_results(outpath.fpath, overwrite=True)


if __name__ == '__main__':
    main()
