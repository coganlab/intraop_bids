import argparse
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from mne_bids import BIDSPath

from PhonemeDatasetBIDS import PhonemeDatasetBIDS
from decoders import DimRedReshape, decodeResultsBIDS

userPath = os.path.expanduser('~')
DERIV_ROOT = Path(userPath) / 'cworkspace' / 'BIDS_1.0_Lexical_µECoG' / 'BIDS' / 'derivatives'
RESULTS_ROOT = DERIV_ROOT / 'decoding'


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
        help='Subject identifier (e.g., "01").'
    )
    parser.add_argument(
        'phonemeIdx',
        type=int,
        help='Phoneme index to decode (1, 2, 3) or -1 for all phonemes.'
    )
    parser.add_argument(
        '--nPhons',
        type=int,
        default=3,
        help='Total number of phonemes in the sequence (default: 3).'
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
        '--chance',
        type=str,
        default='false',
        help='Whether to calculate chance levels of decoding or not (default: "false").'
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


def load_data(bidsRoot, subject, phonemeIdx, nPhons=3, tw=[None, None], **kwargs):
    dataset = PhonemeDatasetBIDS(bidsRoot, subject, phonemeIdx, nPhons, tw,
                                 **kwargs)
    X, y = dataset.get_data()
    return X, y, dataset.twEpoch


def main(bidsRoot, subject, phonemeIdx, nPhons=3, nFolds=20, nIter=50,
         tw=[None, None], compute_chance=False, **kwargs):
    X, y, twEpoch = load_data(bidsRoot, subject, phonemeIdx, nPhons, tw,
                              **kwargs)

    clf = SVC(kernel='rbf', class_weight='balanced')
    pca = DimRedReshape(PCA, n_components=0.8)
    model = make_pipeline(pca, clf)

    decodeHandler = decodeResultsBIDS(
        model=model,
        nFolds=nFolds,
        nIter=nIter,
    )
    decodeHandler.run_decoding(X, y, compute_chance=compute_chance)

    if 'perception' in kwargs.get('description', '').lower():
        phase = 'Perception'
    else:
        phase = 'Production'

    if 'spikeBand' in kwargs.get('suffix', ''):
        band = 'spikeBand'
    else:
        band = 'highgamma'

    if 'zscore' in kwargs.get('description', '').lower():
        normType = 'Zscore'
    elif 'meansub' in kwargs.get('description', '').lower():
        normType = 'MeanSub'
    else:
        normType = 'raw'

    twFmt = ['pre' if twEpoch[0] < 0 else 'post',
             'pre' if twEpoch[1] < 0 else 'post']
    twStr = f"[{twFmt[0]}{abs(twEpoch[0]):.2g},{twFmt[1]}{abs(twEpoch[1]):.2g}]"

    sigStr = 'sig' if 'sig' in kwargs.get('datatype', '') else 'all'

    datatypeStr = 'decode(production)(patientSpecific)'
    if compute_chance:
        datatypeStr += '(chance)'

    outpath = BIDSPath(
        root=RESULTS_ROOT,
        subject=subject,
        datatype=datatypeStr,
        description=f"{band}({phase})({normType})(tw{twStr})({sigStr}Channel)",
        suffix=f"p{'All' if phonemeIdx == -1 else phonemeIdx}",
        extension='.h5',
        check=False,
    )
    outpath.mkdir(exist_ok=True)
    decodeHandler.save_results(outpath.fpath, overwrite=True)


if __name__ == '__main__':
    args = parse_args()

    chance = args.chance.lower() == 'true'

    main(
        bidsRoot=args.bidsRoot,
        subject=args.subject,
        phonemeIdx=args.phonemeIdx,
        nPhons=args.nPhons,
        nFolds=args.nFolds,
        nIter=args.nIter,
        tw=args.tw,
        compute_chance=chance,
        description=args.description,
        suffix=args.suffix,
        extension=args.extension,
        task=args.task,
        datatype=args.datatype,
    )
