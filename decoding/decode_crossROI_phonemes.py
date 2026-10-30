import sys
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
from tqdm import tqdm
from joblib import Parallel, delayed
from mne_bids import BIDSPath

# sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.dataset import PhonemeDatasetBIDS, format_band_str
from utils.decoders import DimRedReshape, NaNImputer3D, decodeResultsBIDS
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


def load_dataset(pt, bids_root, phoneme_idx, tw, elec_rois=None,
                 nan_bad_chans=True, n_phons=3, ps_subjects=None,
                 ps_root=None, **kwargs):

    # can use uECoG data processing now that phoneme level sEEG is available
    return load_align_data(bids_root, pt, phoneme_idx, tw=tw, n_phons=n_phons,
                           elec_rois=elec_rois,
                           nan_bad_chans=nan_bad_chans, ps_subjects=ps_subjects,
                           ps_root=ps_root, **kwargs)


def load_roi_dataset(rois, patients, bids_root, phoneme_idx, tw,
                     nan_bad_chans=True, n_phons=3, **kwargs):
    """Load data for a set of ROIs and patients."""
    # Load data from each ROI separately
    X_list, y_list, y_align_list = [], [], []
    for pt in patients:
        logger.info(f"Loading data for subject {pt}...")
        try:
            X, y, y_align, tw_epoch = load_dataset(
                pt, bids_root, phoneme_idx, tw, n_phons=n_phons,
                nan_bad_chans=nan_bad_chans, elec_rois=rois, **kwargs)
            if 0 in X.shape:
                logger.warning(f"No data found for subject {pt}. Skipping.")
                continue
            X_list.append(X)
            y_list.append(y)
            y_align_list.append(y_align)
        except FileNotFoundError as e:
            logger.warning(f"Data for subject {pt} not found: {e}. Skipping.")

    # Combine data on the channel axis (axis=1)
    X_merge, y_merge, y_align_merge = combine_patient_channels(X_list, y_list, y_align_list)
    return X_merge, y_merge, y_align_merge, tw_epoch


def combine_patient_channels(X_list, y_list, y_align_list):
    """ Combine data (n_trials, ..., n_channels) from multiple patients on the
    channel axis. Requires that the number of trials and labels are the same
    across patients. We use the max number of trials for each label across
    patients and pad with NaNs for patients with fewer trials. Counts are 
    determined based on the alignment labels (y_align) rather than the phoneme 
    labels (y) as alignment labels are tied to higher-level trial sequences."""

    # Sort labels for each patient and get the unique labels
    unique_labels = [np.unique(y_align) for y_align in y_align_list]
    # Get the maximum number of trials for each label
    max_trials_per_label = {}
    for labels in unique_labels:
        for label in labels:
            max_trials_per_label[label] = max(
                max_trials_per_label.get(label, 0),
                *(np.sum(y_align == label) for y_align in y_align_list)
            )
    total_trials = sum(max_trials_per_label.values())

    # Create a new y array with the maximum number of trials for each label
    y_align_merge = np.repeat(list(max_trials_per_label.keys()), list(max_trials_per_label.values()))

    # Pad each patient's data to the maximum number of trials for each label
    X_pad_list, y_pad_list = [], []
    for X, y, y_align in zip(X_list, y_list, y_align_list):
        X_pad = np.zeros((total_trials, *X.shape[1:]))  # Placeholder for padded data
        y_pad = np.zeros(total_trials)  # Placeholder for padded labels
        # For each label, pad the data to the maximum number of trials
        for label in max_trials_per_label:
            n_trials = max_trials_per_label[label]
            X_label = X[y_align == label]
            y_label = y[y_align == label].astype(float)  # Convert to float for NaN padding
            n_pad = n_trials - X_label.shape[0]
            if n_pad > 0:
                X_pad_vec = tuple([(0, n_pad)] + [(0, 0)] * (X_label.ndim - 1))
                X_label = np.pad(X_label, X_pad_vec, mode='constant', constant_values=np.nan)
                y_pad_vec = (0, n_pad)
                y_label = np.pad(y_label, y_pad_vec, mode='constant', constant_values=np.nan)
            # Assign the padded data to the correct indices in the merged array
            X_pad[y_align_merge == label] = X_label
            y_pad[y_align_merge == label] = y_label
        X_pad_list.append(X_pad)
        y_pad_list.append(y_pad)

    # Combine the padded data on the channel axis
    X_merge = np.concatenate(X_pad_list, axis=-1)
    y_merge = merge_on_nan(y_pad_list)

    return X_merge, y_merge, y_align_merge


def merge_on_nan(arrays):
    """Merge multiple arrays of the same shape, taking the first non-NaN value
    along the last axis."""
    stacked = np.stack(arrays, axis=-1)
    merged = np.nanmean(stacked, axis=-1)  # Use nanmean to ignore NaNs
    return merged


def load_subject_data(bids_root, subject, phoneme_idx, n_phons=3,
                      tw=None, ps_root=None, ps_subjects=None,
                      nan_bad_chans=True, elec_rois=None, **kwargs):
    """Load epoched data for a single subject, remapping labels to a common
    phoneme set."""
    if tw is None:
        tw = [None, None]
    if ps_subjects is not None and subject in ps_subjects:
        kwargs['task'] = 'PhonemeSequence'
        bids_root = Path(ps_root).expanduser()
        logger.info(f'Loading PS data for {subject} from {bids_root}')

    dataset = PhonemeDatasetBIDS(bids_root, subject, phoneme_idx, n_phons, tw,
                                 elec_roi=elec_rois,
                                 nan_bad_chans=nan_bad_chans, **kwargs)
    X, y = dataset.get_data()

    label_phons = [dataset.label_dict[lab] for lab in y]
    label_phons = [PS2ARPA.get(p, p) for p in label_phons]
    y = np.array([TARGET_EVENT_ID[p] for p in label_phons])

    return X, y, dataset.twEpoch


def load_align_data(bids_root, subject, phoneme_idx, tw=None, elec_rois=None,
                    nan_bad_chans=True, ps_root=None, ps_subjects=None,
                    n_phons=None, **kwargs):
    """Load subject data and compute alignment labels (phoneme-1 labels)."""

    X, y, tw_epoch = load_subject_data(
        bids_root, subject, phoneme_idx, n_phons, tw, elec_rois=elec_rois,
        ps_root=ps_root, ps_subjects=ps_subjects, nan_bad_chans=nan_bad_chans,
        **kwargs)
    X = X.transpose(0, 2, 1)

    # TODO: make this support different alignment types (only works for sequences right now)
    _, y_align, _ = load_subject_data(
        bids_root, subject, -1, n_phons, tw,
        ps_root=ps_root, ps_subjects=ps_subjects, **kwargs)
    y_align = np.tile(y_align.reshape(-1,n_phons), (n_phons,1,1)).reshape(-1, n_phons, order='F')
    y_align = np.array([''.join(ys.astype(str)) for ys in y_align], dtype=object)  # Convert to string labels for alignment sequences
    # if phoneme_idx == -1:
    #     y_align = np.tile(y_align, (n_phons, 1)).T.flatten()

    return X, y, y_align, tw_epoch


def load_cross_data(rois, patients, bids_root, phoneme_idx, tw=None, random_data=False, **kwargs):
    """Load data for all cross-patient patients."""
    X_list, y_list, y_align_list = [], [], []
    for roi in rois:
        print(f"Loading ROI data from ROI {roi}...")
        X, y, y_align, _ = load_roi_dataset(roi, patients, bids_root,
                                            phoneme_idx, tw, **kwargs)
        if 0 in X.shape:
            print(f"Warning: No data found for ROI {roi}. Skipping.")
            continue
        if random_data:
            print(f"Generating random data for ROI {roi}...")
            X = np.random.rand(*X.shape)
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
            NaNImputer3D(),
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


def build_output_path(cfg, tw_epoch, decode_tw, target_roi, source_rois,
                      results_root, output_suffix=None):
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
    roi_str = ','.join(source_rois) if (isinstance(source_rois, list) or OmegaConf.is_list(cfg.source_rois)) else source_rois

    phoneme_suffix = ('pAll' if cfg.phoneme_idx == -1
                      else f'p{cfg.phoneme_idx}')

    band = format_band_str(cfg.suffix)

    desc = (f"{band}({phase})({norm_type})(alignTW{tw_str(tw_epoch)})"
            f"(decTW{tw_str(decode_tw)})({sig_str}Channel)(crossROIs[{roi_str}])")
    if output_suffix is not None:
        desc += f"({output_suffix})"
    outpath = BIDSPath(
        root=results_root,
        subject=target_roi,
        task=cfg.task,
        datatype='decode(production)(cca)(crossROI)',
        description=desc,
        suffix=phoneme_suffix,
        extension='.h5',
        check=False,
    )
    return outpath


def get_ctx_rois(roi, hemi=None, roi_dict=None):
    # default to both hemispheres if not specified
    if hemi is None:
        hemi = ('lh', 'rh')

    if roi_dict is not None:
        roi = roi_dict.get(roi, roi)

    return {
        f"ctx_{h}_{region}"
        for h in hemi
        for region in roi
    }


@hydra.main(version_base=None, config_path="config",
            config_name="decode_crossROI_phonemes")
def main(cfg: DictConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration:\n{OmegaConf.to_yaml(cfg)}')

    bids_root = Path(cfg.bids_root).expanduser()

    tw = (OmegaConf.to_container(cfg.time_window)
          if cfg.time_window is not None else [None, None])
    decode_tw = list(cfg.decode_tw)

    bids_kwargs = dict(
        description=cfg.description, suffix=cfg.suffix,
        extension=cfg.extension, task=cfg.task, datatype=cfg.datatype)

    # get target ROI labels
    target_rois = get_ctx_rois(cfg.target_rois, cfg.hemi, cfg.roi_dict)
    logger.info(f'Target ROI: specified {cfg.target_rois}, resolved {target_rois}')

    # get source ROI labels (supports multiple source ROI groups)
    source_rois = [get_ctx_rois(roi, cfg.hemi, cfg.roi_dict) for roi in cfg.source_rois]
    logger.info(f'Source ROIs: specified {cfg.source_rois}, resolved {source_rois}')

    # Load target patient data
    logger.info(f'Loading target ROI data from subject(s) {cfg.target_patients} and ROIs {target_rois}...')
    X_tar, y_tar, y_align_tar, tw_epoch = load_roi_dataset(
        target_rois, cfg.target_patients, bids_root, cfg.phoneme_idx, tw, n_phons=cfg.n_phons,
        **bids_kwargs)
    logger.info(f'Target data: X={X_tar.shape}, y={y_tar.shape}, '
                f'y_align={y_align_tar.shape}, tw={tw_epoch}')

    logger.info(f'Loading cross-ROI data from subject(s) {cfg.source_patients} and ROIs {source_rois}...')
    X_src, y_src, y_align_src = load_cross_data(source_rois,
        cfg.source_patients, bids_root, cfg.phoneme_idx, tw,
        n_phons=cfg.n_phons, random_data=cfg.random_data, **bids_kwargs)
    for X_cross, y_cross, y_align_cross in zip(X_src, y_src, y_align_src):
        logger.info(f'Cross-ROI data: X={X_cross.shape}, y={y_cross.shape},'
                    f' y_align={y_align_cross.shape}')

    # Build model
    imputer = NaNImputer3D()
    clf = SVC(kernel=cfg.svm_kernel, class_weight='balanced')
    pca = DimRedReshape(PCA, n_components=cfg.pca_variance_decode)
    model = make_pipeline(imputer, pca, clf)

    logger.info(f'Running cross-patient decoding ({cfg.n_iter} iterations, '
                f'{cfg.n_folds} folds)')
    scores, y_pred_all = aligned_decode(
        X_tar, y_tar, y_align_tar,
        aligner=AlignCCA,
        X_src=X_src, y_src=y_src, y_align_src=y_align_src,
        full_tw=tw_epoch, align_tw=tw_epoch, decode_tw=decode_tw,
        n_folds=cfg.n_folds, n_iter=cfg.n_iter,
        model=model, tar_in_train=cfg.tar_in_train,
        n_comp_align=cfg.pca_variance_align,
        n_comp_decode=cfg.pca_variance_decode,
        n_jobs=-1,
    )

    decode_handler = decodeResultsBIDS(
        model=model, nFolds=cfg.n_folds, nIter=cfg.n_iter,
        scores=scores, y_preds=y_pred_all, scorer='balanced_accuracy')

    if cfg.results_root is not None:
        results_root = Path(cfg.results_root).expanduser()
    else:
        results_root = bids_root.parent / 'decoding(crossROI)'

    outpath = build_output_path(cfg, tw_epoch, decode_tw, cfg.target_rois, cfg.source_rois,
                                results_root, cfg.output_suffix)
    outpath.mkdir(exist_ok=True)
    logger.info(f'Saving results to {outpath.fpath}')
    decode_handler.save_results(outpath.fpath, overwrite=True)


if __name__ == "__main__":
    main()
