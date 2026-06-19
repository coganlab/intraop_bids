"""
MRI visualization/transformation functions
"""

from numpy.typing import ArrayLike
import logging
import re
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import label as ndimage_label
from scipy.spatial import KDTree
import subprocess
import mne
from mne._freesurfer import _read_mri_info, read_ras_mni_t
from mne.transforms import combine_transforms
from mne.viz import Brain
from os import PathLike
from pathlib import Path
import copy

logger = logging.getLogger(__name__)

_GRID_NAME = re.compile(r'^R(\d+)-C(\d+)$')

SUBJECTS_DIR = Path('~/cworkspace/ECoG_Recon').expanduser()
BIDS_ROOT = Path('~/cworkspace/BIDS_1.0_Lexical_µECoG/BIDS').expanduser()

def plot_subj(subj: str, subj_dir: PathLike[str] = SUBJECTS_DIR,
              bids_root: PathLike[str] = BIDS_ROOT,
              elec_pos: ArrayLike | None = None,
              picks: list[str | int] | None = None,
              surface: str = 'pial',
              fig: Brain | None = None,
              elec_kwargs: dict = {},
              brain_kwargs: dict = {},
              units: str = 'mm') -> Brain:
    """
    Plot electrodes on a subject's MRI data.

    Electrode coordinates are assumed to be in scanner RAS (mm), as exported
    from localization on brain.mgz. They are converted to tkrRAS for plotting
    on FreeSurfer surfaces, then scaled to the requested Brain units.
    """
    subj_dir = Path(subj_dir)

    if fig is None:
        # Default to creating a new figure
        # (plotting on existing figure useful for multiple subjects on avg)
        fig = Brain(subj, subjects_dir=subj_dir, surf=surface, hemi='both',
                    units=units, **brain_kwargs)

    # Get default electrode positions and sizes
    default_pos, sizes = _get_elec_pos(subj, bids_root, picks,
                                        return_sizes=True)
    
    # Use default positions if not provided
    if elec_pos is None:
        elec_pos = default_pos
    else:
        elec_pos = np.asarray(elec_pos, dtype=float)

    if units == 'm':
        elec_pos = elec_pos / 1000.0

    # Plot electrodes
    fig = _plot_elecs_on_brain(fig, elec_pos, sizes=sizes, **elec_kwargs)

    return fig


def transform_coords(coords: ArrayLike, transform: np.ndarray) -> np.ndarray:
    """Apply a freesurfer transform (4, 4) to ``(N, 3)`` coordinates."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    homogeneous = np.c_[coords, np.ones(len(coords))]
    return (transform @ homogeneous.T).T[:, :3]


def _get_elec_pos(subj: str, bids_root: PathLike[str],
                  picks: list[str | int] | None = None,
                  return_sizes: bool = False,
                  return_names: bool = False):
    """
    Get electrode positions for the provided subject from the BIDS directory

    If ``return_sizes`` is True, also returns the per-electrode ``size`` column
    (or ``None`` if the column is absent). If ``return_names`` is True, also
    returns the channel name list aligned with ``elec_pos`` rows.
    """
    bids_root = Path(bids_root)

    # Load all electrode data
    elec_files = list(bids_root.glob(
        f'sub-{subj}/**/sub-{subj}*_electrodes.tsv'))
    if not elec_files:
        raise FileNotFoundError(
            f'No electrodes.tsv found for subject {subj}')
    elec_df = pd.read_csv(elec_files[0], sep='\t')

    # Filter only the requested channels
    if picks is not None:
        elec_df = elec_df[elec_df['name'].isin(picks)].reset_index(drop=True)
    else:
        elec_df = elec_df.reset_index(drop=True)

    # Get positions as numpy array
    elec_pos = elec_df[['x', 'y', 'z']].to_numpy()
    names = elec_df['name'].tolist()

    sizes = None
    if return_sizes and 'size' in elec_df.columns:
        # size column is diameter in mm, need to convert to surface area
        diams = elec_df['size'].to_numpy(dtype=float)
        sizes = np.pi * (diams / 2) ** 2

    if return_sizes and return_names:
        return elec_pos, sizes, names
    if return_sizes:
        return elec_pos, sizes
    if return_names:
        return elec_pos, names

    return elec_pos


def _plot_elecs_on_brain(fig: Brain, coords: ArrayLike,
                         sizes: ArrayLike | None = None, **kwargs) -> Brain:
    """ Plot provided coordinates on the brain figure

    ``mne``'s ``Brain.add_foci`` only accepts a single scalar ``scale_factor``
    per call, so to draw electrodes at different sizes we group them by their
    ``size`` value and add each group separately. The effective scale for a
    group is ``scale_factor * size``, i.e. the ``scale_factor`` kwarg acts as a
    base scale that the per-electrode ``size`` column modulates.
    """
    coords = np.asarray(coords, dtype=float)

    # Without per-electrode sizes, fall back to a single scalar scale_factor
    if sizes is None:
        lh_coords = coords[coords[:, 0] < 0]
        rh_coords = coords[coords[:, 0] > 0]
        fig.add_foci(lh_coords, hemi='lh', **kwargs)
        fig.add_foci(rh_coords, hemi='rh', **kwargs)
        return fig

    sizes = np.asarray(sizes, dtype=float)
    base_scale = kwargs.pop('scale_factor', 1.0)
    micro_scale = kwargs.pop('micro_scale_factor', 1.0)

    # Draw one group of foci per distinct size so each gets its own scale
    for size in np.unique(sizes):
        group = coords[sizes == size]
        if size < 1.0:
            group_scale = micro_scale * size
        else:
            group_scale = base_scale * size

        lh_coords = group[group[:, 0] < 0]
        rh_coords = group[group[:, 0] > 0]

        if len(lh_coords):
            fig.add_foci(lh_coords, hemi='lh',
                         scale_factor=group_scale, **kwargs)
        if len(rh_coords):
            fig.add_foci(rh_coords, hemi='rh',
                         scale_factor=group_scale, **kwargs)

    return fig


def plot_subjs_on_avg(subjs: list[str], subj_dir: PathLike[str] = SUBJECTS_DIR,
                      bids_root: PathLike[str] = BIDS_ROOT,
                      avg_subj: str = 'cvs_avg35_inMNI152',
                      method: str = 'spherical',
                      surface: str = 'pial',
                      fig: Brain | None = None,
                      elec_kwargs: dict = {},
                      brain_kwargs: dict = {},
                      units: str = 'mm',
                      visibility: str = 'normal_offset',
                      offset_mm: float | None = None,
                      geometry: str = 'rigid_plate') -> Brain:
    """Plot electrodes from multiple subjects on an average brain surface."""
    subj_dir = Path(subj_dir)

    if avg_subj not in ['fsaverage', 'cvs_avg35_inMNI152']:
        raise ValueError(f'Invalid average subject: {avg_subj}')

    if fig is None:
        avg_kwargs = copy.deepcopy(brain_kwargs)
        avg_kwargs.pop('show', None)
        fig = Brain(avg_subj, subjects_dir=subj_dir, surf=surface, hemi='both',
                    units=units, **avg_kwargs, show=False)

    if method == 'spherical':
        avg_fun = _calc_avg_sph
    elif method == 'talairach':
        avg_fun = _calc_avg_tal
    else:
        raise ValueError(f'Invalid method: {method}')

    sph_kwargs = {}
    if method == 'spherical':
        sph_kwargs = dict(visibility=visibility, offset_mm=offset_mm,
                          geometry=geometry, bids_root=bids_root)

    for subj in subjs:
        elec_pos = _get_elec_pos(subj, bids_root)
        avg_elec_pos = avg_fun(
            elec_pos, subj, avg_subj, subj_dir, **sph_kwargs)
        fig = plot_subj(subj, subj_dir, bids_root, elec_pos=avg_elec_pos,
                        fig=fig, elec_kwargs=elec_kwargs,
                        brain_kwargs=brain_kwargs)
    return fig


def _parse_grid_name(name: str) -> tuple[int, int] | None:
    """Return ``(row, col)`` for ``R#-C#`` channel names, else ``None``."""
    match = _GRID_NAME.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _flood_fill_nans(chan_map: np.ndarray, start_r: int, start_c: int):
    """Find NaN cells reachable from ``(start_r, start_c)`` (8-connected)."""
    n_rows, n_cols = chan_map.shape
    visited = {(start_r, start_c)}
    queue = [(start_r, start_c)]

    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue
            if (0 <= nr < n_rows and 0 <= nc < n_cols
                    and np.isnan(chan_map[nr, nc])):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return visited


def _load_chanmap_bids(bids_root: PathLike[str], subj: str
                       ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Build a chanmap and per-index grid coords from ``channels.tsv``.

    Returns ``(chanmap, chan_coords)`` where ``chan_coords[i]`` is ``(row,
    col)`` for channel index ``i``.  Row order matches ``channels.tsv`` and
    is assumed to align with ``electrodes.tsv`` / iEEG data channel order.
    """
    bids_root = Path(bids_root)
    channels_files = list(bids_root.glob(
        f'sub-{subj}/**/sub-{subj}*_channels.tsv'))
    if not channels_files:
        return None, None

    channels_df = pd.read_csv(channels_files[0], sep='\t')
    chan_coords = []
    for name in channels_df['name'].tolist():
        rc = _parse_grid_name(name)
        if rc is None:
            return None, None
        chan_coords.append(rc)
    chan_coords = np.array(chan_coords)

    n_rows = int(np.max(chan_coords[:, 0])) + 1
    n_cols = int(np.max(chan_coords[:, 1])) + 1
    chanmap = np.full((n_rows, n_cols), np.nan)
    for i, (row, col) in enumerate(chan_coords):
        chanmap[row, col] = i

    if 'electrode_type' in channels_df.columns:
        for i, etype in enumerate(channels_df['electrode_type']):
            if etype == 'macro':
                r, c = chan_coords[i]
                visited = _flood_fill_nans(chanmap, r, c)
                for nr, nc in visited:
                    if (nr, nc) != (r, c):
                        chanmap[nr, nc] = chanmap[r, c]

    return chanmap, chan_coords


def _plate_ids_by_index(chanmap: np.ndarray, chan_coords: np.ndarray,
                        n_electrodes: int) -> np.ndarray:
    """Label electrodes by index using chanmap connected components.

    Electrode index ``i`` is assigned the plate id at grid cell
    ``chan_coords[i]`` from ``channels.tsv``.  Names in ``electrodes.tsv`` are
    not used.
    """
    occupied = ~np.isnan(chanmap)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, _ = ndimage_label(occupied, structure=structure)

    plate_ids = np.full(n_electrodes, -1, dtype=int)
    n = min(n_electrodes, len(chan_coords))
    for i in range(n):
        r, c = int(chan_coords[i, 0]), int(chan_coords[i, 1])
        if (0 <= r < chanmap.shape[0] and 0 <= c < chanmap.shape[1]
                and occupied[r, c]):
            plate_ids[i] = labeled[r, c]
    return plate_ids


def _kabsch_rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Apply a least-squares rigid transform mapping ``source`` -> ``target``."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    centroid_src = source.mean(axis=0)
    centroid_tgt = target.mean(axis=0)
    src_centered = source - centroid_src
    tgt_centered = target - centroid_tgt
    h_mat = src_centered.T @ tgt_centered
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0:
        vt_mat = vt_mat.copy()
        vt_mat[-1] *= -1
        rot = vt_mat.T @ u_mat.T
    return (src_centered @ rot.T) + centroid_tgt


def _rigid_fit_plates(source: np.ndarray, targets: np.ndarray,
                      plate_ids: np.ndarray,
                      min_points: int = 3) -> np.ndarray:
    """Rigidly fit each plate separately; small/unknown plates keep targets."""
    result = targets.copy()
    for pid in np.unique(plate_ids):
        if pid < 0:
            continue
        mask = plate_ids == pid
        if int(mask.sum()) < min_points:
            continue
        result[mask] = _kabsch_rigid_fit(source[mask], targets[mask])
    return result


def _pial_outward_normals(pial_verts: np.ndarray,
                          faces: np.ndarray) -> np.ndarray:
    """Unit outward vertex normals averaged from adjacent face normals."""
    v0 = pial_verts[faces[:, 0]]
    v1 = pial_verts[faces[:, 1]]
    v2 = pial_verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vert_normals = np.zeros_like(pial_verts)
    for i in range(3):
        np.add.at(vert_normals, faces[:, i], face_normals)

    norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    zero = norms[:, 0] == 0
    norms[zero] = 1.0
    vert_normals = vert_normals / norms
    vert_normals[zero] = 0.0
    return vert_normals


def _apply_normal_offset(pos: np.ndarray, vert_idx: np.ndarray,
                         pial_verts: np.ndarray,
                         smoothed_verts: np.ndarray,
                         normals: np.ndarray,
                         offset_mm: float | None = None,
                         min_offset_mm: float = 0.5) -> np.ndarray:
    """Push positions outward along pial vertex normals for pial visibility."""
    pial_at_idx = pial_verts[vert_idx]
    smoothed_at_idx = smoothed_verts[vert_idx]
    normal_at_idx = normals[vert_idx]
    if offset_mm is None:
        gap = np.linalg.norm(pial_at_idx - smoothed_at_idx, axis=1)
        offset = np.maximum(gap, min_offset_mm)
    else:
        offset = np.full(len(pos), offset_mm)
    return pos + normal_at_idx * offset[:, np.newaxis]


def _sph_map_hemi(elec_pos_hemi: np.ndarray, hemi: str,
                  subj_surf_dir: Path, avg_surf_dir: Path,
                  reg_fname_subj: str, reg_fname: str,
                  use_outer: bool) -> tuple[np.ndarray, np.ndarray]:
    """Spherical warp for one hemisphere; returns targets and vertex indices."""
    if use_outer:
        surface = '{}.pial-outer-smoothed'
    else:
        surface = '{}.pial'

    # project to subject smoothed pial surface
    subj_surf, _ = mne.read_surface(subj_surf_dir / surface.format(hemi))
    _, subj_vert_idx = KDTree(subj_surf).query(elec_pos_hemi)

    # spherical registration between subject and average
    sph_subj, _ = mne.read_surface(subj_surf_dir / reg_fname_subj.format(hemi))
    sph_avg, _ = mne.read_surface(avg_surf_dir / reg_fname.format(hemi))
    _, avg_vert_idx = KDTree(sph_avg).query(sph_subj[subj_vert_idx])

    # project to average smoothed pial surface
    avg_surf, _ = mne.read_surface(avg_surf_dir / surface.format(hemi))
    return avg_surf[avg_vert_idx], avg_vert_idx


def _calc_avg_sph(elec_pos: ArrayLike, subj: str, avg_subj: str,
                  subjs_dir: PathLike[str],
                  reg_fname: str = '{}.sphere-outer.reg',
                  visibility: str = 'normal_offset',
                  offset_mm: float | None = None,
                  geometry: str = 'rigid_plate',
                  bids_root: PathLike[str] | None = None) -> np.ndarray:
    """Calculate electrode positions on an average brain via spherical registration.

    Parameters
    ----------
    visibility
        ``'normal_offset'`` pushes foci outward along pial normals so they are
        visible on a pial display; ``'none'`` keeps smoothed-outer coords.
    offset_mm
        Fixed outward offset in mm. ``None`` uses the per-vertex gap between
        pial and smoothed-outer at the mapped vertex index.
    geometry
        ``'rigid_plate'`` preserves native grid geometry via Kabsch fitting per
        chanmap connected component; ``'none'`` keeps independent per-vertex
        warping.
    bids_root
        BIDS root used to load ``channels.tsv`` for chanmap plate segmentation.
        Plate ids are assigned by **row index**: ``elec_pos[i]`` is matched to
        ``channels.tsv`` row ``i`` (same order as iEEG data / chanmap
        localization).  ``electrodes.tsv`` names are not consulted.
    """
    if visibility not in ('none', 'normal_offset'):
        raise ValueError(f'Invalid visibility: {visibility}')
    if geometry not in ('none', 'rigid_plate'):
        raise ValueError(f'Invalid geometry: {geometry}')

    elec_pos = np.asarray(elec_pos, dtype=float)
    if elec_pos.size == 0:
        return np.empty((0, 3))

    if elec_pos.ndim == 1:
        elec_pos = elec_pos.reshape(1, -1)

    subj_dir = Path(subjs_dir) / subj / 'surf'
    avg_dir = Path(subjs_dir) / avg_subj / 'surf'

    reg_fname_subj = copy.deepcopy(reg_fname)
    if avg_subj == 'cvs_avg35_inMNI152':
        reg_fname_subj = reg_fname_subj.removesuffix('.reg') + '-mni.reg'

    use_outer = 'outer' in reg_fname

    plate_ids = None
    if geometry == 'rigid_plate':
        if bids_root is None:
            warnings.warn(
                'rigid_plate geometry requested but bids_root missing; '
                'falling back to per-point warping.',
                stacklevel=2,
            )
            geometry = 'none'
        else:
            chanmap, chan_coords = _load_chanmap_bids(bids_root, subj)
            if chanmap is None:
                warnings.warn(
                    f'Could not build chanmap for subject {subj}; falling '
                    'back to per-point warping.',
                    stacklevel=2,
                )
                geometry = 'none'
            else:
                if len(elec_pos) != len(chan_coords):
                    warnings.warn(
                        f'Subject {subj}: {len(elec_pos)} electrode positions '
                        f'but {len(chan_coords)} channels.tsv rows; plate ids '
                        'assigned by index for the overlapping prefix only.',
                        stacklevel=2,
                    )
                plate_ids = _plate_ids_by_index(
                    chanmap, chan_coords, len(elec_pos))

    pial_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if visibility == 'normal_offset':
        for hemi in ('lh', 'rh'):
            pial_verts, faces = mne.read_surface(avg_dir / f'{hemi}.pial')
            pial_cache[hemi] = (pial_verts, _pial_outward_normals(pial_verts,
                                                                  faces))

    avg_proj_verts = []
    for hemi in ('lh', 'rh'):
        if hemi == 'lh':
            hemi_mask = elec_pos[:, 0] < 0
        else:
            hemi_mask = elec_pos[:, 0] > 0

        elec_pos_hemi = elec_pos[hemi_mask]
        if len(elec_pos_hemi) == 0:
            continue

        targets, vert_idx = _sph_map_hemi(
            elec_pos_hemi, hemi, subj_dir, avg_dir,
            reg_fname_subj, reg_fname, use_outer)

        pos_hemi = targets.copy()
        if geometry == 'rigid_plate' and plate_ids is not None:
            plate_ids_hemi = plate_ids[hemi_mask]
            pos_hemi = _rigid_fit_plates(
                elec_pos_hemi, targets, plate_ids_hemi)

        if visibility == 'normal_offset':
            pial_verts, normals = pial_cache[hemi]
            if use_outer:
                smoothed_fname = f'{hemi}.pial-outer-smoothed'
            else:
                smoothed_fname = f'{hemi}.pial'
            smoothed_verts, _ = mne.read_surface(avg_dir / smoothed_fname)
            pos_hemi = _apply_normal_offset(
                pos_hemi, vert_idx, pial_verts, smoothed_verts,
                normals, offset_mm)

        avg_proj_verts.append(pos_hemi)

    return np.concatenate(avg_proj_verts, axis=0)
        

def read_talxfm(subject: str, subjects_dir: PathLike[str]) -> np.ndarray:
    """Compute the surface-RAS (tkrRAS) -> MNI Talairach transform.

    This is a drop-in replacement for :func:`mne.read_talxfm` that does not
    require ``orig.mgz``/``T1.mgz``. ``recon-all-clinical`` does not write those
    volumes (it produces ``native.mgz`` in a *different* space), but it does
    write ``mri/transforms/talairach.xfm`` and conformed volumes such as
    ``brainmask.mgz`` whose geometry matches the surfaces and the talairach
    transform. We read the RAS->MNI part from ``talairach.xfm`` and derive the
    tkrRAS->scanner-RAS part from the first available conformed volume.

    For standard ``recon-all`` subjects this returns exactly the same transform
    as :func:`mne.read_talxfm` (``orig.mgz`` and ``brainmask.mgz`` share the
    conformed geometry).

    Returns the ``(4, 4)`` affine matrix (in meters).
    """
    subjects_dir = Path(subjects_dir)

    # RAS (scanner, non-zero origin) -> MNI Talairach, from talairach.xfm
    ras_mni_t = read_ras_mni_t(subject, subjects_dir)
    ras_mni_t["trans"][:3, 3] /= 1000.0  # mm -> m

    # tkrRAS ('mri') -> scanner RAS, derived from the conformed volume geometry.
    # orig/T1 for standard recon-all; brainmask for recon-all-clinical.
    mri_dir = subjects_dir / subject / "mri"
    for vol in ("orig.mgz", "T1.mgz", "brainmask.mgz"):
        path = mri_dir / vol
        if path.is_file():
            break
    else:
        raise FileNotFoundError(
            f"No conformed volume (orig/T1/brainmask) found in {mri_dir}")

    _, _, mri_ras_t, _, _ = _read_mri_info(path)
    mri_mni_t = combine_transforms(mri_ras_t, ras_mni_t, "mri", "mni_tal")
    return mri_mni_t["trans"]


def _calc_avg_tal(elec_pos: ArrayLike, subj: str, avg_subj: str,
                  subjs_dir: PathLike[str],
                  surface: str | None = None) -> np.ndarray:
    """ Calculate positions of electrode on average brain surface by using
    talairach registration between subjects then projecting to average brain
    surface
    """
    avg_dir = Path(subjs_dir) / avg_subj / 'surf'

    # use talairach transform to transform electrode positions to average brain
    tal_subj = read_talxfm(subj, subjs_dir)
    tal_avg = read_talxfm(avg_subj, subjs_dir)
    xform = tal_avg @ np.linalg.inv(tal_subj)
    elec_pox_xform = transform_coords(elec_pos, xform)

    if surface is not None: # project to average brain surface
        surface = '{}.'+surface
        avg_proj_verts = []
        for hemi in ['lh', 'rh']:
            if hemi == 'lh':
                elec_pox_xform_hemi = elec_pox_xform[elec_pox_xform[:,0] < 0]
            else:
                elec_pox_xform_hemi = elec_pox_xform[elec_pox_xform[:,0] > 0]

            avg_surf, _ = mne.read_surface(avg_dir / surface.format(hemi))
            tree = KDTree(avg_surf)
            avg_proj_verts_hemi = []
            for coord in elec_pox_xform_hemi:
                closest_vert = tree.query(coord, k=1)[1] # index of closest vertex
                avg_proj_verts_hemi.append(avg_surf[closest_vert][:3]) # x, y, z
            avg_proj_verts_hemi = np.array(avg_proj_verts_hemi)

            if len(avg_proj_verts_hemi) > 0:
                # only add to avg_proj_verts if there are electrodes on this hemisphere
                avg_proj_verts.append(avg_proj_verts_hemi)

        return np.concatenate(avg_proj_verts, axis=0)

    return elec_pox_xform # return in talairach space


def load_channels_tsv(bids_layout, subject):
    """Load a subject's channels.tsv as a DataFrame.

    Parameters
    ----------
    bids_layout : BIDSLayout
        The BIDS layout object for the dataset.
    subject : str
        The subject identifier (without 'sub-' prefix).

    Returns
    -------
    pd.DataFrame
        The channels.tsv contents.
    """
    bids_root = Path(bids_layout.root)
    channels_files = list(bids_root.glob(
        f'sub-{subject}/**/sub-{subject}*_channels.tsv'))
    if not channels_files:
        raise FileNotFoundError(
            f'No channels.tsv found for subject {subject}')

    return pd.read_csv(channels_files[0], sep='\t')

