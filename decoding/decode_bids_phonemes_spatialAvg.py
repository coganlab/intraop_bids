"""Patient-specific phoneme decoding on spatially-averaged epoch data.

Loads epochs from the ``epoch(spatialAvgContact{N})(CAR)`` derivative
produced by ``preprocessing/extract_ieeg_epochs.py`` (with
``spatial_avg.enabled=true``) and runs standard cross-validated
phoneme decoding.  Per-iteration variance comes purely from
re-shuffled CV splits (the averaged data is static for a given
contact size), matching the reference ``aligned_decode_spatialAvg_subsample.py``
behavior.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def build_output_path(cfg, tw_epoch, results_root):
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

    datatype_str = ('decode(production)(patientSpecific)'
                    '(spatialAvgSubsample)')
    if cfg.compute_chance:
        datatype_str += '(chance)'

    phoneme_suffix = ('pAll' if cfg.phoneme_idx == -1
                      else f'p{cfg.phoneme_idx}')

    contact_str = f'contact{int(cfg.contact_size)}'

    outpath = BIDSPath(
        root=results_root,
        subject=cfg.patient,
        datatype=datatype_str,
        description=(f"{band}({phase})({norm_type})"
                     f"(tw{tw_str})({sig_str}Channel)"
                     f"({contact_str})"),
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
            config_name='decode_phonemes_spatial_avg')
def main(cfg: DictConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration:\n{OmegaConf.to_yaml(cfg)}')

    bids_root = Path(cfg.bids_root).expanduser()

    tw = (OmegaConf.to_container(cfg.time_window)
          if cfg.time_window is not None else [None, None])

    logger.info(f'Loading spatial-average epochs for subject {cfg.patient}, '
                f'contact_size={cfg.contact_size}...')
    dataset = PhonemeDatasetBIDS(
        bids_root, cfg.patient, cfg.phoneme_idx, cfg.n_phons, tw,
        description=cfg.description, suffix=cfg.suffix,
        extension=cfg.extension, task=cfg.task, datatype=cfg.datatype)
    X, y = dataset.get_data()
    logger.info(f'Loaded data: X={X.shape}, y={y.shape}, '
                f'tw={dataset.twEpoch}')

    imputer = NaNImputer3D()
    pca = DimRedReshape(PCA, n_components=cfg.pca_variance)
    clf = SVC(kernel=cfg.svm_kernel, class_weight='balanced')
    model = make_pipeline(imputer, pca, clf)

    n_iter = int(cfg.n_subsample_iters)
    scores = np.zeros(n_iter)
    y_preds = np.zeros((n_iter, len(y)))

    for j in tqdm(range(n_iter), desc='Spatial-avg CV iters'):
        cv = _make_cv(X, y, cfg.n_folds)
        score, y_pred = _decode_iter(model, X, y, cv,
                                     compute_chance=cfg.compute_chance)
        scores[j] = score
        y_preds[j] = y_pred
        logger.info(f'Iter {j+1}/{n_iter}: bal_acc={score:.4f}')

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

    outpath = build_output_path(cfg, dataset.twEpoch, results_root)
    outpath.mkdir(exist_ok=True)
    logger.info(f'Saving results to {outpath.fpath}')
    decoder.save_results(outpath.fpath, overwrite=True)


if __name__ == '__main__':
    main()
