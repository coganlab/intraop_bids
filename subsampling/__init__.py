"""Channel-subsampling strategies for high-density iEEG arrays.

Submodules
----------
chanmap_utils : helpers for translating BIDS chanmap grids and channel names
poisson_disk  : pitch subsampling via Poisson-disk sampling
coverage      : grid (subarray) coverage subsampling
spatial_avg   : spatial averaging of contacts to simulate larger contact sizes
"""

from . import chanmap_utils, coverage, poisson_disk, spatial_avg

__all__ = [
    'chanmap_utils',
    'coverage',
    'poisson_disk',
    'spatial_avg',
]
