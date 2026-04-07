import argparse
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from mne_bids import BIDSPath

from PhonemeDatasetBIDS import PhonemeDatasetBIDS
from decoders import DimRedReshape, decodeResultsBIDS
from alignment.AlignCCA import AlignCCA
from cross_pt_decoders import crossPtDecoder_sepDimRed, crossPtDecoder_twSepAlign


userPath = os.path.expanduser('~')
DERIV_ROOT = Path(userPath) /'cworkspace/BIDS_1.0_Lexical_µECoG/BIDS/derivatives'
RESULTS_ROOT = DERIV_ROOT / 'decoding'

PS_ROOT = Path(userPath) / 'cworkspace/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS/derivatives/epoch(phonemeLevel)(CAR)'

# PS_PTS = ['S14', 'S22', 'S23', 'S26', 'S33', 'S62']
PS_PTS = ['S14','S22','S23','S26','S33','S39','S58','S62']
# LEX_PTS = ['S41', 'S45', 'S47', 'S51', 'S53', 'S55', 'S56', 'S63', 'S67', 'S73', 'S74', 'S75', 'S76']
LEX_PTS = ['S41','S45','S47','S55','S56','S63','S67','S73','S74','S75','S76','S78']
# LEX_PTS = ['S41','S45','S73','S74','S75','S76','S78']
ALIGNER = AlignCCA
TW_DEC = [-0.5, 0.5]

N_COMP_ALIGN = 0.9  # variance explained for alignment PCA
N_COMP_DECODE = 0.8  # variance explained for decoding PCA

PS2ARPA = {'a': 'AA', 'ae': 'EH', 'i': 'IY', 'u': 'UH', 'b': 'B', 'p': 'P',
           'v': 'V', 'g': 'G', 'k': 'K'}
TARGET_EVENT_ID = {'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AY': 5, 'B': 6, 'D': 7,
                   'EH': 8, 'EY': 9, 'F': 10, 'G': 11, 'HH': 12, 'IH': 13,
                   'IY': 14, 'JH': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19,
                   'OW': 20, 'P': 21, 'R': 22, 'S': 23, 'T': 24, 'UH': 25,
                   'V': 26, 'W': 27, 'Z': 28}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode phoneme information from BIDS formatted data."
    )
    parser.add_argument(
        'bidsRoot',
        type=str,
        help='Path to BIDS root directory (likely derivatives section).'
    )
    parser.add_argument(
        'subject',
        type=str,
        help='Target subject identifier (e.g., "S73").'
    )
    parser.add_argument(
        'phonemeIdx',
        type=int,
        help='Phoneme index to decode (1, 2, 3) or -1 for all phonemes.'
    )
    parser.add_argument(
        'poolTask',
        type=str,
        default='all',
        help='Which tasks to pool for cross-patient alignment (default: "all").',
    )
    parser.add_argument(
        '--nFolds',
        type=int,
        default=20,
        help='Number of cross-validation folds (default: 20).'
    )
    parser.add_argument(
        '--nIter',
        type=int,
        default=50,
        help='Number of decoding iterations (default: 50).'
    )
    parser.add_argument(
        '--tw',
        type=float,
        nargs=2,
        default=[None, None],
        help='Time window (in seconds) to use for decoding, e.g.,'
             '--tw -0.5 1.0. Default is full epoch.'
    )
    parser.add_argument(
        '--description',
        type=str,
        default='productionMeanSub',
        help='Description field to match in BIDSPath (default: "productionMeanSub").'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='highgamma',
        help='Suffix of the BIDS file to load (default: "highgamma").'
    )
    parser.add_argument(
        '--extension',
        type=str,
        default='.fif',
        help='File extension of the BIDS file to load (default: ".fif").'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='PhonemeSequence',
        help='Task label in BIDS (default: "phonemeSequence").'
    )
    parser.add_argument(
        '--datatype',
        type=str,
        default='epoch(band)(power)',
        help='Data type in BIDS (default: "epoch(band)(power)").'
    )
    return parser.parse_args()


def load_subject_data(bidsRoot, subject, phonemeIdx, nPhons=3, tw=[None, None], **kwargs):
    if subject in PS_PTS:
        kwargs['task'] = 'PhonemeSequence'
        bidsRoot = PS_ROOT
        print(f"Loading data for patient {subject} from {bidsRoot}")
    dataset = PhonemeDatasetBIDS(bidsRoot, subject, phonemeIdx, nPhons, tw,
                                 **kwargs)
    X, y = dataset.get_data()

    # remap labels to common set to account for any differences arising from
    # different tasks or missing trials during preprocessing
    labelPhons = [dataset.label_dict[lab] for lab in y]
    labelPhons = [PS2ARPA.get(p, p) for p in labelPhons]  # map PS to ARPA if needed
    y = np.array([TARGET_EVENT_ID[p] for p in labelPhons])

    return X, y, dataset.twEpoch

def run_fold(train_idx, test_idx, X, y, y_align, model, aligner):
    """Run one CV fold and return predictions + test indices."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_align_train, _ = y_align[train_idx], y_align[test_idx]

    # fold-specific setup
    if aligner is not None:
        model.set_align_labels(y_align_train)

    # fit + predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return test_idx, y_pred

def aligned_decode(X_tar, y_tar, y_align_tar=None, aligner=None,
                   X_src=None, y_src=None, y_align_src=None,
                   full_tw=[-1,1], align_tw=[-1,1], decode_tw=[-0.5,0.5],
                   n_folds=10, n_iter=10, model=None, tar_in_train=True,
                   n_jobs=-1):
    # default model
    if model is None:
        clf = SVC(kernel='rbf', class_weight='balanced')
        model = make_pipeline(
            DimRedReshape(PCA, n_components=N_COMP_DECODE),
            clf
        )
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True)
    try:
        _ = list(cv.split(X_tar, y_tar))
    except ValueError:
        cv = KFold(n_splits=n_folds, shuffle=True)

    if y_align_tar is None:
        y_align_tar = y_tar
    if X_src is not None and y_align_src is None:
        y_align_src = y_src

    if X_src is not None and y_src is not None:
        pool = True
        if aligner is None:
            model = crossPtDecoder_sepDimRed(list(zip(X_src, y_src, y_align_src)),
                                             model, tar_in_train=tar_in_train, n_comp=N_COMP_ALIGN)
        else:
            model = crossPtDecoder_twSepAlign(list(zip(X_src, y_src, y_align_src)),
                                              model, aligner, full_tw, align_tw, decode_tw,
                                              tar_in_train=tar_in_train, n_comp=N_COMP_ALIGN)
    else:
        pool = False

    y_pred_all = np.zeros((n_iter, len(y_tar)))
    scores = np.zeros(n_iter)

    for i in tqdm(range(n_iter), desc='Decoding iterations'):
        if pool:
            # parallelize across folds
            results = Parallel(n_jobs=n_jobs)(
                delayed(run_fold)(train_idx, test_idx,
                                  X_tar, y_tar, y_align_tar,
                                  clone(model), aligner)
                for train_idx, test_idx in cv.split(X_tar, y_tar)
            )
            y_pred_iter = np.zeros(len(y_tar))
            for test_idx, y_pred in results:
                y_pred_iter[test_idx] = y_pred
        else:
            # fall back to sklearn’s built-in parallel CV
            y_pred_iter = cross_val_predict(model, X_tar, y_tar, cv=cv, n_jobs=n_jobs)

        scores[i] = balanced_accuracy_score(y_tar, y_pred_iter)
        print(f"Iteration: {i+1} - balanced acuracy: {scores[i]:.4f}", flush=True)
        y_pred_all[i, :] = y_pred_iter

    return scores, y_pred_all


def load_cross_data(bidsRoot, subjects, phonemeIdx, tw=[None, None], **kwargs):
    X_list, y_list, y_align_list = [], [], []
    for subj in subjects:
        X, y, y_align, _ = load_align_data(bidsRoot, subj, phonemeIdx, tw,
                                           **kwargs)
        
        X_list.append(X)
        y_list.append(y)
        y_align_list.append(y_align)  # using same labels for alignment for now
    return X_list, y_list, y_align_list


def load_align_data(bidsRoot, subject, phonemeIdx, tw=[None, None],
                     **kwargs):
    nPhons = get_nPhons(subject)
    X, y, twEpoch = load_subject_data(bidsRoot, subject, phonemeIdx, nPhons, tw,
                                      **kwargs)
    X = X.transpose(0, 2, 1)

    # # simply align by phoneme across positions
    # y_align = np.array(y).squeeze()

    # # can use full label based alignment for within-task if desired (won't work across tasks)
    # y_align = np.tile(y.reshape(-1,nPhons), (nPhons,1,1)).reshape(-1, nPhons, order='F')

    # always align by phoneme 1 labels
    _, y_align, _ = load_subject_data(bidsRoot, subject, 1, nPhons, tw,
                                      **kwargs)
    if phonemeIdx == -1:
        # need to repeat p1 labels for alignment if using all phonemes
        # transpose + flatten is to make sure that first phoneme is repeated
        # for all phoneme positions of a single trial then for the next trial,
        # (e.g. 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, ...), rathen than
        # interleaved across trials (e.g. 1, 2, 3, ..., 1, 2, 3, ...
        y_align = np.tile(y_align, (nPhons,1)).T.flatten()
    return X, y, y_align, twEpoch


def get_nPhons(subject):
    if subject in PS_PTS:
        return 3
    # elif subject in LEX_PTS:
    #     return 5
    # else:
    #     raise ValueError(f"Subject {subject} not recognized. Please update subject lists within this script.")
    else:
        return 5


def remap_labels(labels, src_dict, tar_dict):
    labels = np.array(labels).squeeze()
    labels_remap = np.zeros_like(labels)
    for src_id, src_phon in src_dict.items():
        for tar_id, tar_phon in tar_dict.items():
            if src_phon == tar_phon:
                labels_remap[labels == src_id] = tar_id
    return labels_remap


def swap_kv_dict(d):
    return dict((v, k) for k, v in d.items())


def main(bidsRoot, subject, phonemeIdx, poolTask, nFolds=20, nIter=50,
         tw=[None, None], **kwargs):
    # data for target patient
    X_tar, y_tar, y_align_tar, twEpoch = load_align_data(
        bidsRoot, subject, phonemeIdx, tw, **kwargs)

    # data across all other patients for alignment
    if poolTask.lower() in ['phoneme', 'phonemesequence', 'phoneme_sequence']:
        other_subjects = sorted([s for s in PS_PTS if s != subject])
    elif poolTask.lower() in ['lexical', 'lexicalrepeat', 'lexical_repeat']:
        other_subjects = sorted([s for s in LEX_PTS if s != subject])
    elif poolTask.lower() in ['all', 'both']:
        other_subjects = sorted([s for s in PS_PTS + LEX_PTS if s != subject])
    else:
        raise ValueError(f"Unrecognized poolTask option: {poolTask}. "
                         "Please choose from 'phoneme', 'lexical', or 'all'.")
    # other_subjects = sorted([s for s in PS_PTS + LEX_PTS if s != subject])
    X_src, y_src, y_align_src = load_cross_data(
        bidsRoot, other_subjects, phonemeIdx, tw, **kwargs)

    clf = SVC(kernel='rbf', class_weight='balanced')
    pca = DimRedReshape(PCA, n_components=N_COMP_DECODE)
    model = make_pipeline(pca, clf)

    scores, y_pred_all = aligned_decode(
        X_tar, y_tar, y_align_tar,
        aligner=ALIGNER,
        X_src=X_src, y_src=y_src, y_align_src=y_align_src,
        full_tw=twEpoch,
        align_tw=twEpoch,
        decode_tw=TW_DEC,
        n_folds=nFolds,
        n_iter=nIter,
        model=model,
        tar_in_train=True,
        n_jobs=1,
    )

    decodeHandler = decodeResultsBIDS(
        model=model,
        nFolds=nFolds,
        nIter=nIter,
        scores=scores,
        y_preds=y_pred_all,
        scorer='balanced_accuracy'
    )
    

    if 'perception' in kwargs.get('description', '').lower():
        phase = 'Perception'
    else:
        phase = 'Production'

    if 'zscore' in kwargs.get('description', '').lower():
        normType = 'Zscore'
    elif 'meansub' in kwargs.get('description', '').lower():
        normType = 'MeanSub'
    else:
        normType = 'raw'

    twFmtAlign = ['pre' if twEpoch[0] < 0 else 'post',
                 'pre' if twEpoch[1] < 0 else 'post']
    twStrAlign = f"[{twFmtAlign[0]}{abs(twEpoch[0]):.2g},{twFmtAlign[1]}{abs(twEpoch[1]):.2g}]"
    twFmtDec = ['pre' if TW_DEC[0] < 0 else 'post',
                'pre' if TW_DEC[1] < 0 else 'post']
    twStrDec = f"[{twFmtDec[0]}{abs(TW_DEC[0]):.2g},{twFmtDec[1]}{abs(TW_DEC[1]):.2g}]"

    sigStr = 'sig' if 'sig' in kwargs.get('datatype', '') else 'all'

    ptStr = ','.join(other_subjects)

    outpath = BIDSPath(
        root=RESULTS_ROOT,
        subject=subject,
        task=kwargs.get('task', 'lexical'),
        datatype='decode(production)(cca)(crossPatientTask)',
        description=f"highgamma({phase})({normType})(alignTW{twStrAlign})(decTW{twStrDec})({sigStr}Channel)(pts[{ptStr}])",
        suffix=f"p{'All' if phonemeIdx == -1 else phonemeIdx}",
        extension='.h5',
        check=False,
    )
    outpath.mkdir(exist_ok=True)
    decodeHandler.save_results(outpath.fpath, overwrite=True)


if __name__ == '__main__':
    args = parse_args()
    main(
        bidsRoot=args.bidsRoot,
        subject=args.subject,
        phonemeIdx=args.phonemeIdx,
        poolTask=args.poolTask,
        nFolds=args.nFolds,
        nIter=args.nIter,
        tw=args.tw,
        description=args.description,
        suffix=args.suffix,
        extension=args.extension,
        task=args.task,
        datatype=args.datatype,
    )
