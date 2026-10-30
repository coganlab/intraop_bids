"""Project feature values onto a 3D brain surface.

Loads pre-computed z-scored feature data from BIDS derivatives and renders
electrode-level values on a subject or average brain surface.  For average-
space projection, uses standard pial spherical registration (sphere.reg)
with outer-smoothed surface indexing to avoid sulcus splitting on micro-ECoG
grids.
"""
import csv
import sys
import logging
from collections import OrderedDict

import mne
import numpy as np
import matplotlib.cm as cm
import hydra
from omegaconf import DictConfig, OmegaConf
from mne.viz import Brain
from mne_bids import BIDSPath
from pathlib import Path
from bids.layout import BIDSLayout

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.utils import snake2camel
# loc/mri.py holds the shared electrode plotting/projection code
# (used interactively in loc/brainplot.ipynb).
from visualization.loc import mri as mri_loc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_subject_files(patient, subjects_dir, average,
                           atlas=None, atlas_radius=None):
    """Verify all required neuroanatomical files exist before processing.

    Raises ``FileNotFoundError`` with a list of every missing path.
    """
    missing = []

    subj_path = subjects_dir / patient
    if not subj_path.is_dir():
        raise FileNotFoundError(
            f"Patient directory not found: {subj_path}")

    for hemi in ('lh', 'rh'):
        for name in (f'{hemi}.pial-outer-smoothed',
                     f'{hemi}.sphere.reg'):
            p = subj_path / 'surf' / name
            if not p.exists():
                missing.append(str(p))

    elec_file = (subj_path / 'elec_recon'
                 / f'{patient}_elec_locations_RAS_brainshifted.txt')
    if not elec_file.exists():
        missing.append(str(elec_file))

    tpl_path = subjects_dir / average
    if not tpl_path.is_dir():
        raise FileNotFoundError(
            f"Template brain directory not found: {tpl_path}")

    for hemi in ('lh', 'rh'):
        for name in (f'{hemi}.pial', f'{hemi}.pial-outer-smoothed',
                     f'{hemi}.sphere.reg'):
            p = tpl_path / 'surf' / name
            if not p.exists():
                missing.append(str(p))

    if atlas is not None and atlas_radius is not None:
        csv_base = (f'{patient}_elec_location_radius_{atlas_radius}mm'
                    f'_aparc{atlas}+aseg.mgz')
        csv_path = subj_path / 'elec_recon' / (csv_base + '.csv')
        csv_bs = subj_path / 'elec_recon' / (csv_base + '_brainshifted.csv')
        if not csv_path.exists() and not csv_bs.exists():
            missing.append(str(csv_path))

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n  " + "\n  ".join(missing))


# ---------------------------------------------------------------------------
# Electrode I/O
# ---------------------------------------------------------------------------

def load_electrode_positions(patient, subjects_dir):
    """Load electrode RAS positions from elec_recon.

    Returns an ``OrderedDict`` mapping channel name to ``(x, y, z)`` in
    millimetres (FreeSurfer surface-RAS, matching surface vertex coords).
    """
    elec_file = (subjects_dir / patient / 'elec_recon'
                 / f'{patient}_elec_locations_RAS_brainshifted.txt')
    elecs = OrderedDict()
    with open(elec_file, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            parts = row[0].split(" ")
            name = "".join(parts[0:2])
            coords = tuple(float(n) for n in parts[2:5])
            elecs[name] = coords
    return elecs


# ---------------------------------------------------------------------------
# Average-space projection
# ---------------------------------------------------------------------------

def project_electrodes_spherical(patient, subjects_dir,
                                 electrode_positions, average,
                                 bids_root=None,
                                 visibility='normal_offset',
                                 offset_mm=None,
                                 geometry='rigid_plate'):
    """Project electrode positions to template space via spherical registration.

    Thin wrapper around :func:`mri._calc_avg_sph` (the shared implementation
    in ``loc/mri.py``) that preserves the channel-name -> coordinate mapping.

    ``_calc_avg_sph`` splits electrodes by hemisphere (``x < 0`` -> ``lh``,
    ``x > 0`` -> ``rh``) and returns the projected coordinates concatenated as
    ``[lh..., rh...]``, dropping any electrode with ``x == 0``.  We reproduce
    that ordering here so the returned dict stays aligned with the projected
    positions.

    Parameters
    ----------
    patient : str
        Subject ID (e.g. ``'S14'``).
    subjects_dir : Path
        Root of the ECoG_Recon directory tree.
    electrode_positions : OrderedDict
        Channel name -> ``(x, y, z)`` in mm (surface-RAS).
    average : str
        Template brain directory name (e.g. ``'cvs_avg35_inMNI152'``).
    bids_root
        BIDS dataset root for chanmap-based rigid plate fitting.  Plate
        segmentation uses ``channels.tsv`` row order; ``coords`` must be in
        the same order (index ``i`` in ``coords`` ↔ row ``i`` in
        ``channels.tsv``).
    visibility : str
        ``'normal_offset'`` or ``'none'`` (see ``mri._calc_avg_sph``).
    offset_mm : float, optional
        Fixed outward normal offset in mm; ``None`` uses auto per-vertex gap.
    geometry : str
        ``'rigid_plate'`` or ``'none'`` (see ``mri._calc_avg_sph``).

    Returns
    -------
    OrderedDict
        Channel name -> ``(x, y, z)`` projected onto the template surface,
        in mm.
    """
    names = list(electrode_positions.keys())
    coords = np.array(list(electrode_positions.values()), dtype=float)
    if len(coords) == 0:
        return OrderedDict()

    proj = mri_loc._calc_avg_sph(
        coords, patient, average, subjects_dir,
        bids_root=bids_root,
        visibility=visibility,
        offset_mm=offset_mm,
        geometry=geometry,
    )

    # Mirror _calc_avg_sph's hemisphere ordering to realign names with coords.
    lh_idx = np.where(coords[:, 0] < 0)[0]
    rh_idx = np.where(coords[:, 0] > 0)[0]
    order = np.concatenate([lh_idx, rh_idx]).astype(int)

    projected = OrderedDict()
    for pos, idx in zip(proj, order):
        projected[names[idx]] = tuple(pos)
    return projected


# ---------------------------------------------------------------------------
# Region labels
# ---------------------------------------------------------------------------

def get_electrode_region_labels(patient, subjects_dir, ch_names,
                                atlas='.a2009s', radius=10):
    """Look up anatomical region labels for electrodes from atlas CSVs.

    Returns an ``OrderedDict`` mapping each channel name to its most-
    probable grey-matter region label (string).
    """
    from ieeg.viz.mri import subject_to_info, gen_labels

    info = subject_to_info(patient, str(subjects_dir))
    labels = gen_labels(info, patient, str(subjects_dir), atlas,
                        picks=info.ch_names)

    region_map = OrderedDict()
    for ch in ch_names:
        region_map[ch] = labels.get(ch, 'unknown')
    return region_map


def save_region_labels_tsv(filepath, ch_names, region_labels, ch_values):
    """Write a TSV file mapping each electrode to its region and value."""
    with open(filepath, 'w') as f:
        f.write('channel\tregion\tfeature_value\n')
        for ch, val in zip(ch_names, ch_values):
            region = region_labels.get(ch, 'unknown')
            f.write(f'{ch}\t{region}\t{val:.6f}\n')


def is_white_matter_label(label):
    """Return ``True`` if an atlas region label denotes white matter.

    Used to honour the ``rm_wm`` option now that the in-house ``plot_subj``
    no longer performs the filtering for us.  Matching is heuristic and based
    on the conventions of the ``aparc``/``aseg`` labels returned by
    ``ieeg.viz.mri.gen_labels`` (e.g. ``Left-Cerebral-White-Matter``,
    ``wm-lh-...``).
    """
    if not label:
        return False
    lab = str(label).lower()
    return 'white' in lab or lab.startswith('wm')


# ---------------------------------------------------------------------------
# Feature-epoch helpers (unchanged)
# ---------------------------------------------------------------------------

def load_feature_epochs(patient, bids_layout, task_period, feature,
                        reference):
    """Load a z-scored feature epoch file from BIDS derivatives.

    Parameters
    ----------
    patient : str
        Subject ID.
    bids_layout : BIDSLayout
        BIDS dataset layout.
    task_period : str
        Task-period name (e.g. ``'perception'``).
    feature : str
        Feature name in snake_case (e.g. ``'high_gamma'``).
    reference : str
        Reference scheme used during epoch extraction.
    """
    feature_base_dir = (Path(bids_layout.root) / 'derivatives'
                        / f'epoch({reference})')

    for datatype in ['epoch(band)(power)(sig)', 'epoch(band)(power)']:
        feature_path = BIDSPath(
            root=feature_base_dir,
            subject=patient,
            task=bids_layout.get_tasks()[0] if bids_layout.get_tasks()
                 else None,
            datatype=datatype,
            suffix=snake2camel(feature),
            extension=".fif",
            description=f"{task_period}Zscore",
            check=False
        )
        print(feature_path)
        matches = feature_path.match()
        if matches:
            return mne.read_epochs(matches[0], preload=True)

    return None


def compute_channel_values(epochs, time_window=None):
    """Compute a single value per channel by averaging across epochs and time.

    Parameters
    ----------
    epochs : mne.Epochs
        Feature epochs (trials x channels x time).
    time_window : tuple or None
        ``(tmin, tmax)`` in seconds.  ``None`` uses the full epoch.

    Returns
    -------
    np.ndarray
        Shape ``(n_channels,)`` with the mean feature value per channel.
    """
    if time_window is not None:
        epochs = epochs.copy().crop(tmin=time_window[0], tmax=time_window[1])

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    return np.nanmean(data, axis=(0, 2))  # mean over epochs and time


def values_to_colors(values, cmap_name='hot'):
    """Map channel values to RGBA colors via a matplotlib colormap."""
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if np.isclose(vmin, vmax):
        norm_vals = np.full_like(values, 0.5)
    else:
        norm_vals = (values - vmin) / (vmax - vmin)

    cmap = cm.get_cmap(cmap_name)
    return [cmap(v) for v in norm_vals]


def values_to_sizes(values, size_min=0.2, size_max=1.5):
    """Map channel values to electrode marker sizes."""
    abs_vals = np.abs(values)
    vmin, vmax = np.nanmin(abs_vals), np.nanmax(abs_vals)
    if np.isclose(vmin, vmax):
        return [0.5 * (size_min + size_max)] * len(values)

    norm = (abs_vals - vmin) / (vmax - vmin)
    return [size_min + n * (size_max - size_min) for n in norm]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def build_brain(subject, subjects_dir, cfg):
    """Create a ``Brain`` figure for *subject* using the configured surface.

    Mirrors the ``Brain`` construction in ``loc/mri.py`` (``units='m'``,
    white background) and falls back to flat cortex shading when curvature
    data is missing for a template.
    """
    try:
        return Brain(subject, subjects_dir=str(subjects_dir),
                     cortex='low_contrast', alpha=cfg.brain_alpha,
                     background='white', surf=cfg.surface, hemi=cfg.hemi,
                     units='m', show=False)
    except Exception:
        logger.warning('Curvature data missing for %s; using flat cortex '
                       'shading.', subject)
        return Brain(subject, subjects_dir=str(subjects_dir),
                     cortex='ivory', alpha=cfg.brain_alpha,
                     background='white', surf=cfg.surface, hemi=cfg.hemi,
                     units='m', show=False)


def add_feature_foci(brain, coords_mm, colors, sizes, hemi='both'):
    """Place per-electrode coloured foci on a ``Brain`` figure.

    This is the value-coloured counterpart to ``mri._plot_elecs_on_brain``:
    electrodes are routed to a hemisphere by the sign of their x coordinate
    (matching that module's convention) and drawn one at a time so each can
    carry its own colour and ``scale_factor``.

    *coords_mm* are in millimetres (FreeSurfer surface-RAS) and are converted
    to metres to match ``Brain(units='m')``.  *colors*, *sizes* and
    *coords_mm* must share the same ordering.
    """
    coords = np.asarray(coords_mm, dtype=float)
    if len(coords) == 0:
        return
    coords_m = coords / 1000.0

    for p, c, s in zip(coords_m, colors, sizes):
        elec_hemi = 'lh' if p[0] < 0 else 'rh'
        if hemi in ('lh', 'rh') and elec_hemi != hemi:
            continue
        if len(c) == 4:
            brain.add_foci(p, hemi=elec_hemi, color=c[:3],
                           scale_factor=s, alpha=c[3])
        else:
            brain.add_foci(p, hemi=elec_hemi, color=c, scale_factor=s)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="config",
            config_name="project_HG_brain")
def main(cfg: DictConfig):
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f'Missing configuration keys: {missing_keys}')

    logger.info(f'Running with configuration:\n{OmegaConf.to_yaml(cfg)}')

    cfg.bids_root = Path(cfg.bids_root).expanduser()
    subjects_dir = Path(cfg.subjects_dir).expanduser()

    # --- Validate neuroanatomical files up front ---
    validate_subject_files(
        cfg.patient, subjects_dir, cfg.average,
        atlas=cfg.atlas if cfg.label_electrodes else None,
        atlas_radius=cfg.atlas_radius if cfg.label_electrodes else None,
    )

    bids_layout = BIDSLayout(root=cfg.bids_root)

    task_periods = list(cfg.task_periods)
    features = list(cfg.features)

    time_window = None
    if cfg.time_window is not None:
        time_window = tuple(cfg.time_window)

    fig_outpath = (Path(bids_layout.root) / 'derivatives' / 'figs'
                   / 'brainProjection')
    fig_outpath.mkdir(parents=True, exist_ok=True)

    # --- Load electrode positions from ECoG_Recon ---
    all_elec_positions = load_electrode_positions(cfg.patient, subjects_dir)
    logger.info(f'Loaded {len(all_elec_positions)} electrode positions '
                f'from elec_recon')

    # --- Region labels (once per subject, reused across features) ---
    region_labels = None
    if cfg.label_electrodes:
        try:
            region_labels = get_electrode_region_labels(
                cfg.patient, subjects_dir,
                list(all_elec_positions.keys()),
                atlas=cfg.atlas, radius=cfg.atlas_radius)
            logger.info('Electrode region labels:')
            for ch, region in region_labels.items():
                logger.info(f'  {ch}: {region}')
        except Exception as e:
            logger.warning(f'Could not load region labels: {e}')

    for tp_name in task_periods:
        for feat_name in features:
            logger.info(f'##### {tp_name} / {feat_name} #####')

            epochs = load_feature_epochs(cfg.patient, bids_layout,
                                         tp_name, feat_name, cfg.reference)
            if epochs is None:
                logger.warning(f'No data found for {tp_name} '
                               f'{feat_name}. Skipping.')
                continue

            ch_values = compute_channel_values(epochs, time_window)

            # Match epoch channels to elec_recon electrode positions
            ch_value_map = dict(zip(epochs.ch_names, ch_values))
            active_elecs = OrderedDict()
            active_values = []
            for ch_name, pos in all_elec_positions.items():
                if ch_name in ch_value_map:
                    active_elecs[ch_name] = pos
                    active_values.append(ch_value_map[ch_name])

            active_values = np.array(active_values)
            if len(active_values) == 0:
                logger.warning('No epoch channels matched elec_recon '
                               'positions. Skipping.')
                continue

            colors = values_to_colors(active_values, cfg.cmap)
            sizes = values_to_sizes(
                active_values,
                size_min=cfg.electrode_size.min,
                size_max=cfg.electrode_size.max,
            )

            logger.info(f'Value range: [{active_values.min():.3f}, '
                        f'{active_values.max():.3f}]')
            logger.info(f'Channels matched: {len(active_elecs)} of '
                        f'{len(epochs.ch_names)} epoch channels')

            names = list(active_elecs.keys())
            color_map = dict(zip(names, colors))
            size_map = dict(zip(names, sizes))

            if cfg.projection == 'average':
                # --- Spherical registration projection (loc/mri.py) ---
                projected = project_electrodes_spherical(
                    cfg.patient, subjects_dir, active_elecs, cfg.average,
                    bids_root=Path(bids_layout.root))

                plot_names = list(projected.keys())
                plot_coords = np.array(list(projected.values()), dtype=float)
                brain = build_brain(cfg.average, subjects_dir, cfg)

            elif cfg.projection == 'subject':
                plot_names = list(names)
                plot_coords = np.array(list(active_elecs.values()),
                                       dtype=float)

                # rm_wm previously handled inside ieeg's plot_subj; replicate
                # by dropping electrodes whose atlas label is white matter.
                if cfg.rm_wm:
                    if region_labels is None:
                        logger.warning('rm_wm=True but no region labels are '
                                       'available; keeping all electrodes.')
                    else:
                        keep = [not is_white_matter_label(
                                    region_labels.get(n, 'unknown'))
                                for n in plot_names]
                        n_drop = len(keep) - sum(keep)
                        if n_drop:
                            logger.info(f'rm_wm: dropping {n_drop} '
                                        f'white-matter electrodes.')
                        plot_names = [n for n, k in zip(plot_names, keep) if k]
                        plot_coords = plot_coords[np.array(keep, dtype=bool)]

                brain = build_brain(cfg.patient, subjects_dir, cfg)
            else:
                raise ValueError(
                    f'Unknown projection type: {cfg.projection}')

            plot_colors = [color_map[n] for n in plot_names]
            plot_sizes = [size_map[n] for n in plot_names]
            add_feature_foci(brain, plot_coords, plot_colors, plot_sizes,
                             hemi=cfg.hemi)

            # --- Save region-label TSV ---
            if region_labels is not None:
                tsv_fname = (f'sub-{cfg.patient}_task-{cfg.task}'
                             f'_desc-{tp_name}_{feat_name}'
                             f'_regions.tsv')
                save_region_labels_tsv(
                    fig_outpath / tsv_fname,
                    list(active_elecs.keys()),
                    region_labels,
                    active_values)
                logger.info(f'Saved region labels: '
                            f'{fig_outpath / tsv_fname}')

            # --- Save view screenshots ---
            for view_cfg in cfg.views:
                mne.viz.set_3d_view(brain, azimuth=view_cfg.azimuth,
                                    elevation=view_cfg.elevation)

                fname = (f'sub-{cfg.patient}_task-{cfg.task}'
                         f'_desc-{tp_name}_{feat_name}'
                         f'_view-{view_cfg.name}.{cfg.output_format}')
                out_file = fig_outpath / fname

                brain.save_image(str(out_file))
                logger.info(f'Saved: {out_file}')

            brain.close()

    logger.info('Brain projection complete!')


if __name__ == "__main__":
    main()
