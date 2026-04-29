import sys
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from mne_bids import BIDSPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.dataset import PhonemeDatasetBIDS
from utils.decoders import DimRedReshape, decodeResultsBIDS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def build_output_path(cfg, twEpoch, results_root):
    """Construct a BIDSPath for saving decoding results."""
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

    tw_fmt = ['pre' if twEpoch[0] < 0 else 'post',
              'pre' if twEpoch[1] < 0 else 'post']
    tw_str = (f"[{tw_fmt[0]}{abs(twEpoch[0]):.2g},"
              f"{tw_fmt[1]}{abs(twEpoch[1]):.2g}]")

    sig_str = 'sig' if 'sig' in cfg.datatype else 'all'

    datatype_str = 'decode(production)(patientSpecific)'
    if cfg.compute_chance:
        datatype_str += '(chance)'

    phoneme_suffix = ('pAll' if cfg.phoneme_idx == -1
                      else f'p{cfg.phoneme_idx}')

    outpath = BIDSPath(
        root=results_root,
        subject=cfg.patient,
        datatype=datatype_str,
        description=(f"{band}({phase})({norm_type})"
                     f"(tw{tw_str})({sig_str}Channel)"),
        suffix=phoneme_suffix,
        extension='.h5',
        check=False,
    )
    return outpath


@hydra.main(version_base=None, config_path="config",
            config_name="decode_phonemes")
def main(cfg: DictConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration:\n{OmegaConf.to_yaml(cfg)}')

    bids_root = Path(cfg.bids_root).expanduser()

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

    pca = DimRedReshape(PCA, n_components=cfg.pca_variance)
    clf = SVC(kernel=cfg.svm_kernel, class_weight='balanced')
    model = make_pipeline(pca, clf)

    decoder = decodeResultsBIDS(
        model=model, nFolds=cfg.n_folds, nIter=cfg.n_iter)
    logger.info(f'Running decoding ({cfg.n_iter} iterations, '
                f'{cfg.n_folds} folds, chance={cfg.compute_chance})...')
    decoder.run_decoding(X, y, compute_chance=cfg.compute_chance)

    if cfg.results_root is not None:
        results_root = Path(cfg.results_root).expanduser()
    else:
        results_root = bids_root.parent / 'decoding'

    outpath = build_output_path(cfg, dataset.twEpoch, results_root)
    outpath.mkdir(exist_ok=True)
    logger.info(f'Saving results to {outpath.fpath}')
    decoder.save_results(outpath.fpath, overwrite=True)


if __name__ == "__main__":
    main()
