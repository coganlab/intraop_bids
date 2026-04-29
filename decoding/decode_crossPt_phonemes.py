import sys
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
from tqdm import tqdm
from joblib import Parallel, delayed
from mne_bids import BIDSPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.dataset import PhonemeDatasetBIDS
from utils.decoders import DimRedReshape, decodeResultsBIDS
from utils.alignment import AlignCCA
from utils.crossPt_decoders import (crossPtDecoder_sepDimRed,
                                     crossPtDecoder_twSepAlign)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PS2ARPA = {
    'a': 'AA', 'ae': 'EH', 'i': 'IY', 'u': 'UH',
    'b': 'B', 'p': 'P', 'v': 'V', 'g': 'G', 'k': 'K',
}
TARGET_EVENT_ID = {
    'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AY': 5, 'B': 6, 'D': 7,
    'EH': 8, 'EY': 9, 'F': 10, 'G': 11, 'HH': 12, 'IH': 13,
    'IY': 14, 'JH': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19,
    'OW': 20, 'P': 21, 'R': 22, 'S': 23, 'T': 24, 'UH': 25,
    'V': 26, 'W': 27, 'Z': 28,
}


def get_n_phons(subject, ps_subjects):
    if subject in ps_subjects:
        return 3
    return 5


def load_subject_data(bids_root, subject, phoneme_idx, n_phons=3,
                      tw=None, ps_root=None, ps_subjects=None, **kwargs):
    """Load epoched data for a single subject, remapping labels to a common
    phoneme set."""
    if tw is None:
        tw = [None, None]
    if ps_subjects is not None and subject in ps_subjects:
        kwargs['task'] = 'PhonemeSequence'
        bids_root = Path(ps_root).expanduser()
        logger.info(f'Loading PS data for {subject} from {bids_root}')

    dataset = PhonemeDatasetBIDS(bids_root, subject, phoneme_idx, n_phons, tw,
                                 **kwargs)
    X, y = dataset.get_data()

    label_phons = [dataset.label_dict[lab] for lab in y]
    label_phons = [PS2ARPA.get(p, p) for p in label_phons]
    y = np.array([TARGET_EVENT_ID[p] for p in label_phons])

    return X, y, dataset.twEpoch


def load_align_data(bids_root, subject, phoneme_idx, tw=None,
                    ps_root=None, ps_subjects=None, **kwargs):
    """Load subject data and compute alignment labels (phoneme-1 labels)."""
    n_phons = get_n_phons(subject, ps_subjects or [])
    X, y, tw_epoch = load_subject_data(
        bids_root, subject, phoneme_idx, n_phons, tw,
        ps_root=ps_root, ps_subjects=ps_subjects, **kwargs)
    X = X.transpose(0, 2, 1)

    _, y_align, _ = load_subject_data(
        bids_root, subject, 1, n_phons, tw,
        ps_root=ps_root, ps_subjects=ps_subjects, **kwargs)
    if phoneme_idx == -1:
        y_align = np.tile(y_align, (n_phons, 1)).T.flatten()

    return X, y, y_align, tw_epoch


def load_cross_data(bids_root, subjects, phoneme_idx, tw=None,
                    ps_root=None, ps_subjects=None, **kwargs):
    """Load data for all cross-patient subjects."""
    X_list, y_list, y_align_list = [], [], []
    for subj in subjects:
        X, y, y_align, _ = load_align_data(
            bids_root, subj, phoneme_idx, tw,
            ps_root=ps_root, ps_subjects=ps_subjects, **kwargs)
        X_list.append(X)
        y_list.append(y)
        y_align_list.append(y_align)
    return X_list, y_list, y_align_list


def run_fold(train_idx, test_idx, X, y, y_align, model, aligner):
    """Run one CV fold and return predictions + test indices."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, _ = y[train_idx], y[test_idx]
    y_align_train = y_align[train_idx]

    if aligner is not None:
        model.set_align_labels(y_align_train)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return test_idx, y_pred


def aligned_decode(X_tar, y_tar, y_align_tar=None, aligner=None,
                   X_src=None, y_src=None, y_align_src=None,
                   full_tw=None, align_tw=None, decode_tw=None,
                   n_folds=10, n_iter=10, model=None, tar_in_train=True,
                   n_comp_align=0.9, n_comp_decode=0.8, n_jobs=-1):
    if full_tw is None:
        full_tw = [-1, 1]
    if align_tw is None:
        align_tw = [-1, 1]
    if decode_tw is None:
        decode_tw = [-0.5, 0.5]

    if model is None:
        clf = SVC(kernel='rbf', class_weight='balanced')
        model = make_pipeline(
            DimRedReshape(PCA, n_components=n_comp_decode), clf)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True)
    try:
        list(cv.split(X_tar, y_tar))
    except ValueError:
        cv = KFold(n_splits=n_folds, shuffle=True)

    if y_align_tar is None:
        y_align_tar = y_tar
    if X_src is not None and y_align_src is None:
        y_align_src = y_src

    if X_src is not None and y_src is not None:
        pool = True
        if aligner is None:
            model = crossPtDecoder_sepDimRed(
                list(zip(X_src, y_src, y_align_src)),
                model, tar_in_train=tar_in_train, n_comp=n_comp_align)
        else:
            model = crossPtDecoder_twSepAlign(
                list(zip(X_src, y_src, y_align_src)),
                model, aligner, full_tw, align_tw, decode_tw,
                tar_in_train=tar_in_train, n_comp=n_comp_align)
    else:
        pool = False

    y_pred_all = np.zeros((n_iter, len(y_tar)))
    scores = np.zeros(n_iter)

    for i in tqdm(range(n_iter), desc='Decoding iterations'):
        if pool:
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_fold)(train_idx, test_idx, X_tar, y_tar,
                                  y_align_tar, clone(model), aligner)
                for train_idx, test_idx in cv.split(X_tar, y_tar))
            y_pred_iter = np.zeros(len(y_tar))
            for test_idx, y_pred in results:
                y_pred_iter[test_idx] = y_pred
        else:
            from sklearn.model_selection import cross_val_predict
            y_pred_iter = cross_val_predict(model, X_tar, y_tar, cv=cv,
                                            n_jobs=n_jobs)

        scores[i] = balanced_accuracy_score(y_tar, y_pred_iter)
        logger.info(f'Iteration {i + 1}/{n_iter} - '
                     f'balanced accuracy: {scores[i]:.4f}')
        y_pred_all[i, :] = y_pred_iter

    return scores, y_pred_all


def build_output_path(cfg, tw_epoch, decode_tw, other_subjects, results_root):
    """Construct a BIDSPath for saving cross-patient decoding results."""
    description = cfg.description
    phase = ('Perception' if 'perception' in description.lower()
             else 'Production')

    if 'zscore' in description.lower():
        norm_type = 'Zscore'
    elif 'meansub' in description.lower():
        norm_type = 'MeanSub'
    else:
        norm_type = 'raw'

    def tw_str(tw):
        fmt = ['pre' if tw[0] < 0 else 'post',
               'pre' if tw[1] < 0 else 'post']
        return f"[{fmt[0]}{abs(tw[0]):.2g},{fmt[1]}{abs(tw[1]):.2g}]"

    sig_str = 'sig' if 'sig' in cfg.datatype else 'all'
    pt_str = ','.join(other_subjects)

    phoneme_suffix = ('pAll' if cfg.phoneme_idx == -1
                      else f'p{cfg.phoneme_idx}')

    outpath = BIDSPath(
        root=results_root,
        subject=cfg.patient,
        task=cfg.task,
        datatype='decode(production)(cca)(crossPatientTask)',
        description=(f"{cfg.suffix}({phase})({norm_type})"
                     f"(alignTW{tw_str(tw_epoch)})"
                     f"(decTW{tw_str(decode_tw)})"
                     f"({sig_str}Channel)"
                     f"(pts[{pt_str}])"),
        suffix=phoneme_suffix,
        extension='.h5',
        check=False,
    )
    return outpath


@hydra.main(version_base=None, config_path="config",
            config_name="decode_crosspt_phonemes")
def main(cfg: DictConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration:\n{OmegaConf.to_yaml(cfg)}')

    bids_root = Path(cfg.bids_root).expanduser()
    ps_root = Path(cfg.ps_root).expanduser()
    ps_subjects = list(cfg.ps_subjects)
    lex_subjects = list(cfg.lex_subjects)

    tw = (OmegaConf.to_container(cfg.time_window)
          if cfg.time_window is not None else [None, None])
    decode_tw = list(cfg.decode_tw)

    bids_kwargs = dict(
        description=cfg.description, suffix=cfg.suffix,
        extension=cfg.extension, task=cfg.task, datatype=cfg.datatype)

    # Load target patient data
    logger.info(f'Loading target data for {cfg.patient}...')
    X_tar, y_tar, y_align_tar, tw_epoch = load_align_data(
        bids_root, cfg.patient, cfg.phoneme_idx, tw,
        ps_root=str(ps_root), ps_subjects=ps_subjects, **bids_kwargs)
    logger.info(f'Target data: X={X_tar.shape}, y={y_tar.shape}, '
                f'tw={tw_epoch}')

    # Determine which other subjects to pool
    pool_task = cfg.pool_task.lower()
    if pool_task in ['phoneme', 'phonemesequence', 'phoneme_sequence']:
        other_subjects = sorted(s for s in ps_subjects if s != cfg.patient)
    elif pool_task in ['lexical', 'lexicalrepeat', 'lexical_repeat']:
        other_subjects = sorted(s for s in lex_subjects if s != cfg.patient)
    elif pool_task in ['all', 'both']:
        other_subjects = sorted(
            s for s in ps_subjects + lex_subjects if s != cfg.patient)
    else:
        raise ValueError(
            f"Unrecognized pool_task: {cfg.pool_task}. "
            "Choose from 'phoneme', 'lexical', or 'all'.")

    logger.info(f'Loading cross-patient data for {other_subjects}...')
    X_src, y_src, y_align_src = load_cross_data(
        bids_root, other_subjects, cfg.phoneme_idx, tw,
        ps_root=str(ps_root), ps_subjects=ps_subjects, **bids_kwargs)

    # Build model
    clf = SVC(kernel=cfg.svm_kernel, class_weight='balanced')
    pca = DimRedReshape(PCA, n_components=cfg.pca_variance_decode)
    model = make_pipeline(pca, clf)

    logger.info(f'Running cross-patient decoding ({cfg.n_iter} iterations, '
                f'{cfg.n_folds} folds, {len(other_subjects)} source patients)')
    scores, y_pred_all = aligned_decode(
        X_tar, y_tar, y_align_tar,
        aligner=AlignCCA,
        X_src=X_src, y_src=y_src, y_align_src=y_align_src,
        full_tw=tw_epoch, align_tw=tw_epoch, decode_tw=decode_tw,
        n_folds=cfg.n_folds, n_iter=cfg.n_iter,
        model=model, tar_in_train=True,
        n_comp_align=cfg.pca_variance_align,
        n_comp_decode=cfg.pca_variance_decode,
        n_jobs=1,
    )

    decode_handler = decodeResultsBIDS(
        model=model, nFolds=cfg.n_folds, nIter=cfg.n_iter,
        scores=scores, y_preds=y_pred_all, scorer='balanced_accuracy')

    if cfg.results_root is not None:
        results_root = Path(cfg.results_root).expanduser()
    else:
        results_root = bids_root.parent / 'decoding'

    outpath = build_output_path(cfg, tw_epoch, decode_tw, other_subjects,
                                results_root)
    outpath.mkdir(exist_ok=True)
    logger.info(f'Saving results to {outpath.fpath}')
    decode_handler.save_results(outpath.fpath, overwrite=True)


if __name__ == "__main__":
    main()
