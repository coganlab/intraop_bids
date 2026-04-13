"""Preprocessing utility package.

Submodules
----------
config : Hydra config loading helpers
dataloaders : BIDS data loading
feature_extraction : Band power / Hilbert envelope extraction
preprocessing : Shared preprocessing pipeline steps
referencing : Re-referencing schemes (CAR, WM, Laplacian)
stats : Outlier detection, bad trial removal, significance testing
"""


def snake2camel(snake_str):
    """Convert a snake_case string to camelCase."""
    str_parts = snake_str.split('_')
    result = str_parts[0]
    if len(str_parts) > 1:
        result += ''.join([part.capitalize() for part in str_parts[1:]])
    return result


__all__ = [
    'snake2camel',
    'config',
    'dataloaders',
    'feature_extraction',
    'preprocessing',
    'referencing',
    'stats',
]
