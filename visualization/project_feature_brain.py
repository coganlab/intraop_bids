"""Project feature values onto a 3D brain surface.

Loads pre-computed z-scored feature data from BIDS derivatives and renders
electrode-level values on a subject or average brain surface.  For average-
space projection, uses spherical surface registration via sphere-outer.reg
to avoid distortion from Talairach affine transforms on micro-ECoG grids.
"""
import csv
import sys
import logging
from collections import OrderedDict

import mne
import numpy as np
import matplotlib.cm as cm
import nibabel.freesurfer as fs
import hydra
from omegaconf import DictConfig, OmegaConf
from mne.viz import Brain
from mne_bids import BIDSPath
from pathlib import Path
from bids.layout import BIDSLayout
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.utils import snake2camel

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
                     f'{hemi}.sphere-outer.reg'):
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
        for name in (f'{hemi}.pial', f'{hemi}.sphere-outer.reg'):
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


def split_by_hemisphere(electrode_positions):
    """Split electrodes into left/right dicts.

    Uses the channel-name prefix (``L`` / ``R``); falls back to the sign
    of the x-coordinate when neither prefix matches.
    """
    left = OrderedDict()
    right = OrderedDict()
    for name, pos in electrode_positions.items():
        if name.startswith('L'):
            left[name] = pos
        elif name.startswith('R'):
            right[name] = pos
        elif pos[0] < 0:
            left[name] = pos
        else:
            right[name] = pos
    return left, right


# ---------------------------------------------------------------------------
# Spherical registration projection
# ---------------------------------------------------------------------------

def project_electrodes_spherical(patient, subjects_dir,
                                 electrode_positions, average):
    """Project electrode positions to template space via spherical registration.

    For each hemisphere the steps are:

    1. Find the nearest vertex on the subject's ``pial-outer-smoothed``
       surface for every electrode.
    2. Look up the corresponding vertex in the subject's
       ``sphere-outer.reg`` (same vertex indexing).
    3. Find the nearest vertex on the *template's* ``sphere-outer.reg``
       to that sphere coordinate.
    4. Return the template's ``pial`` vertex position at that index.

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

    Returns
    -------
    OrderedDict
        Channel name -> ``(x, y, z)`` projected onto the template pial
        surface, in mm.
    """
    left, right = split_by_hemisphere(electrode_positions)
    projected = OrderedDict()

    for hemi, elecs in [('lh', left), ('rh', right)]:
        if not elecs:
            continue

        subj_pial_verts, _ = fs.read_geometry(
            str(subjects_dir / patient / 'surf'
                / f'{hemi}.pial-outer-smoothed'))
        subj_sphere_verts, _ = fs.read_geometry(
            str(subjects_dir / patient / 'surf'
                / f'{hemi}.sphere-outer.reg'))

        tpl_sphere_verts, _ = fs.read_geometry(
            str(subjects_dir / average / 'surf'
                / f'{hemi}.sphere-outer.reg'))
        tpl_pial_verts, _ = fs.read_geometry(
            str(subjects_dir / average / 'surf' / f'{hemi}.pial'))

        subj_pial_tree = cKDTree(subj_pial_verts)
        tpl_sphere_tree = cKDTree(tpl_sphere_verts)

        elec_coords = np.array(list(elecs.values()))

        _, subj_vert_idx = subj_pial_tree.query(elec_coords)
        subj_sphere_coords = subj_sphere_verts[subj_vert_idx]

        _, tpl_vert_idx = tpl_sphere_tree.query(subj_sphere_coords)
        tpl_positions = tpl_pial_verts[tpl_vert_idx]

        for i, name in enumerate(elecs.keys()):
            projected[name] = tuple(tpl_positions[i])

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
    return np.mean(data, axis=(0, 2))  # mean over epochs and time


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

def _add_electrodes(brain, positions_mm, colors, sizes, hemi):
    """Place electrode foci on a ``Brain`` figure for one hemisphere.

    *positions_mm* values are in millimetres (FreeSurfer surface-RAS);
    they are converted to metres to match ``Brain(units='m')``.
    """
    pos_array = np.array(list(positions_mm.values()))
    if len(pos_array) == 0:
        return
    pos_m = pos_array / 1000.0

    for i, p in enumerate(pos_m):
        c = colors[i]
        s = sizes[i]
        if len(c) == 4:
            brain.add_foci(p, hemi=hemi, color=c[:3],
                           scale_factor=s, alpha=c[3])
        else:
            brain.add_foci(p, hemi=hemi, color=c, scale_factor=s)


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

    from ieeg.viz.mri import plot_subj

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

            if cfg.projection == 'average':
                # --- Spherical registration projection ---
                projected = project_electrodes_spherical(
                    cfg.patient, subjects_dir, active_elecs, cfg.average)

                left_proj, right_proj = split_by_hemisphere(projected)

                color_map = dict(zip(active_elecs.keys(), colors))
                size_map = dict(zip(active_elecs.keys(), sizes))
                left_colors = [color_map[n] for n in left_proj]
                left_sizes = [size_map[n] for n in left_proj]
                right_colors = [color_map[n] for n in right_proj]
                right_sizes = [size_map[n] for n in right_proj]

                try:
                    brain = Brain(
                        cfg.average, subjects_dir=str(subjects_dir),
                        cortex='low_contrast', alpha=cfg.brain_alpha,
                        background='white', surf=cfg.surface,
                        hemi=cfg.hemi, units='m', show=False)
                except Exception:
                    logger.warning(
                        'Curvature data missing for template; '
                        'using flat cortex shading.')
                    brain = Brain(
                        cfg.average, subjects_dir=str(subjects_dir),
                        cortex='ivory', alpha=cfg.brain_alpha,
                        background='white', surf=cfg.surface,
                        hemi=cfg.hemi, units='m', show=False)

                if left_proj and cfg.hemi != 'rh':
                    _add_electrodes(brain, left_proj,
                                    left_colors, left_sizes, 'lh')
                if right_proj and cfg.hemi != 'lh':
                    _add_electrodes(brain, right_proj,
                                    right_colors, right_sizes, 'rh')

            elif cfg.projection == 'subject':
                brain = plot_subj(
                    cfg.patient,
                    subj_dir=str(subjects_dir),
                    no_wm=cfg.rm_wm,
                    surface=cfg.surface,
                    hemi=cfg.hemi,
                    color=colors,
                    size=sizes,
                    background='white',
                    transparency=cfg.brain_alpha,
                    show=False,
                )
            else:
                raise ValueError(
                    f'Unknown projection type: {cfg.projection}')

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
