"""Shared Hydra configuration loading utilities.

Provides helpers for resolving config names (strings) into OmegaConf objects
from YAML files on disk.  Used by all Hydra entry-point scripts to normalise
``task_periods`` and ``features`` lists.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(item, config_type, config_root=None):
    """Load a sub-config YAML by name.

    Parameters
    ----------
    item : str or DictConfig
        If a string, treated as the stem of a YAML file to load.
    config_type : str
        Subdirectory under *config_root* (e.g. ``"cfg_task_periods"``).
    config_root : Path or None
        Root config directory.  Defaults to ``<caller's dir>/config``.
    """
    if isinstance(item, str):
        if config_root is None:
            config_root = Path(__file__).resolve().parent.parent / 'config'
        else:
            config_root = Path(config_root)
        config_file = config_root / config_type / f'{item}.yaml'
        return OmegaConf.load(config_file)
    return item


def normalize_cfg_items(item, config_type, config_root=None):
    """Normalise a Hydra override value into a list of config objects.

    Handles bare strings, ``"[a,b]"`` bracket notation, OmegaConf lists and
    dicts so that callers always receive a uniform ``list[DictConfig]``.
    """
    if isinstance(item, str):
        item = item.strip()
        if item.startswith('[') and item.endswith(']'):
            return normalize_cfg_items(OmegaConf.create(item), config_type,
                                       config_root)
        return [load_config(item, config_type, config_root)]

    if isinstance(item, DictConfig):
        return [item]

    if OmegaConf.is_list(item) or isinstance(item, (list, tuple)):
        return [load_config(x, config_type, config_root) for x in item]

    if OmegaConf.is_dict(item):
        return [item]

    return [item]
