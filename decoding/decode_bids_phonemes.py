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

from utils.dataset import PhonemeDatasetBIDS, build_output_path
from utils.decoders import DimRedReshape, NaNImputer3D, decodeResultsBIDS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


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
            config_name="decode_phonemes")
def main(cfg: DictConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration:\n{OmegaConf.to_yaml(cfg)}')

    bids_root = Path(cfg.bids_root).expanduser()

    tw = (OmegaConf.to_container(cfg.time_window)
          if cfg.time_window is not None else [None, None])

    # get ROI labels
    if cfg.roi is not None:
        rois = get_ctx_rois(cfg.roi, cfg.hemi, cfg.roi_dict)
        logger.info(f'Target ROI: specified {cfg.roi}, resolved {rois}')
    else:
        rois = None
        logger.info('No ROI specified.')

    logger.info(f'Loading data for subject {cfg.patient}...')
    dataset = PhonemeDatasetBIDS(
        bids_root, cfg.patient, cfg.phoneme_idx, cfg.n_phons, tw,
        elec_roi=rois, nan_bad_chans=True,
        description=cfg.description, suffix=cfg.suffix,
        extension=cfg.extension, task=cfg.task, datatype=cfg.datatype)
    X, y = dataset.get_data()
    logger.info(f'Loaded data: X={X.shape}, y={y.shape}, '
                f'tw={dataset.twEpoch}')

    imputer = NaNImputer3D()
    pca = DimRedReshape(PCA, n_components=cfg.pca_variance)
    clf = SVC(kernel=cfg.svm_kernel, class_weight='balanced')
    model = make_pipeline(imputer, pca, clf)

    decoder = decodeResultsBIDS(
        model=model, nFolds=cfg.n_folds, nIter=cfg.n_iter)
    logger.info(f'Running decoding ({cfg.n_iter} iterations, '
                f'{cfg.n_folds} folds, chance={cfg.compute_chance})...')
    decoder.run_decoding(X, y, compute_chance=cfg.compute_chance)

    if cfg.results_root is not None:
        results_root = Path(cfg.results_root).expanduser()
    else:
        results_root = bids_root.parent / 'decoding'

    outpath = build_output_path(
                    results_root, 'decode(patientSpecific)', cfg.patient,
                    cfg.task, cfg.description, cfg.datatype, cfg.suffix, tw,
                    cfg.phoneme_idx, output_suffix=cfg.output_suffix)
    outpath.mkdir(exist_ok=True)
    logger.info(f'Saving results to {outpath.fpath}')
    decoder.save_results(outpath.fpath, overwrite=True)


if __name__ == "__main__":
    main()
