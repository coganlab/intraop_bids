"""
MRI visualization/transformation functions
"""

from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import subprocess
import mne
from mne._freesurfer import _read_mri_info, read_ras_mni_t
from mne.transforms import combine_transforms
from mne.viz import Brain
from os import PathLike
from pathlib import Path
import copy

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
              units: str = 'm') -> Brain:
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
                  return_sizes: bool = False):
    """
    Get electrode positions for the provided subject from the BIDS directory

    If ``return_sizes`` is True, also returns the per-electrode ``size`` column
    (or ``None`` if the column is absent).
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

    if return_sizes:
        sizes = (elec_df['size'].to_numpy(dtype=float)
                 if 'size' in elec_df.columns else None)
        return elec_pos, sizes

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

    # Draw one group of foci per distinct size so each gets its own scale
    for size in np.unique(sizes):
        group = coords[sizes == size]
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
                      units: str = 'm') -> Brain:
    """ Plot electrodes from multiple subjects on a average brain surface
    """
    
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

    for subj in subjs:
        elec_pos = _get_elec_pos(subj, bids_root)
        avg_elec_pos = avg_fun(elec_pos, subj, avg_subj, subj_dir)
        fig = plot_subj(subj, subj_dir, bids_root, elec_pos=avg_elec_pos,
                        fig=fig, elec_kwargs=elec_kwargs,
                        brain_kwargs=brain_kwargs)
    return fig


def _calc_avg_sph(elec_pos: ArrayLike, subj: str, avg_subj: str,
                  subjs_dir: PathLike[str],
                  reg_fname: str = '{}.sphere-outer.reg') -> np.ndarray:
    """ Calculate positions of electrode on average brain surface via
    spherical registration between subjects
    """
    elec_pos = np.asarray(elec_pos, dtype=float)
    subj_dir = Path(subjs_dir) / subj / 'surf'
    avg_dir = Path(subjs_dir) / avg_subj / 'surf'

    reg_fname_subj = copy.deepcopy(reg_fname)
    if avg_subj == 'cvs_avg35_inMNI152':
        reg_fname_subj = reg_fname_subj.removesuffix('.reg') + '-mni.reg'
    
    avg_proj_verts = []
    for hemi in ['lh', 'rh']:
        if hemi == 'lh':
            elec_pos_hemi = elec_pos[elec_pos[:,0] < 0]
        else:
            elec_pos_hemi = elec_pos[elec_pos[:,0] > 0]

        if 'outer' in reg_fname:
            surface = '{}.pial-outer-smoothed'
        else:
            surface = '{}.pial'

        # Project electrode positions to subject brain surface
        surf, _ = mne.read_surface(subj_dir / surface.format(hemi))
        tree = KDTree(surf)
        proj_verts_hemi = []
        for coord in elec_pos_hemi:
            closest_vert = tree.query(coord, k=1)[1]
            proj_verts_hemi.append(closest_vert)
        proj_verts_hemi = np.array(proj_verts_hemi)

        # Project from subject to average using spherical registration
        sph_subj, _ = mne.read_surface(subj_dir / reg_fname_subj.format(hemi))
        sph_avg, _ = mne.read_surface(avg_dir / reg_fname.format(hemi))
        tree = KDTree(sph_avg)
        avg_sph_verts = []
        for coord in proj_verts_hemi:
            subj_sph_vert = sph_subj[coord]
            closest_avg_vert = tree.query(subj_sph_vert, k=1)[1]
            avg_sph_verts.append(closest_avg_vert)
        avg_sph_verts = np.array(avg_sph_verts)

        # Project to average brain surface from average spherical space
        avg_surf, _ = mne.read_surface(avg_dir / surface.format(hemi))
        avg_proj_verts_hemi = []
        for coord in avg_sph_verts:
            # don't need to query closest vertex, just look up loc in avg surf
            avg_proj_verts_hemi.append(avg_surf[coord])
        avg_proj_verts_hemi = np.array(avg_proj_verts_hemi)

        # only add to avg_proj_verts if there are electrodes on this hemisphere
        if len(avg_proj_verts_hemi) > 0:
            avg_proj_verts.append(avg_proj_verts_hemi)

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

