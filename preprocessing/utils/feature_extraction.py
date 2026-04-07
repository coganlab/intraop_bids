"""Feature extraction utilities for IEEG data.

This module provides functions for extracting features from neural data,
including bandpass filtering, power extraction, and Hilbert transform-based
envelope detection.

Functions:
    extract_band: Extract band power with filtering and smoothing
    extract_band_hilbert: Extract band envelope using Hilbert transform
"""

from typing import List, Literal
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mne import BaseEpochs
from ieeg.timefreq import gamma


def validate_freqs(frange: List[float]):
    if len(frange) != 2 or frange[0] >= frange[1]:
        raise ValueError(f"frange must be [low_freq, high_freq] with low_freq "
                         f"< high_freq, got {frange}")
    if frange[0] <= 0:
        raise ValueError(f"frange must be positive, got {frange}")


def extract_band(
    epochs: BaseEpochs,
    frange: List[float] = [300, 1000],
    power: Literal['abs', 'square'] = 'abs',
    smooth_ms: float = 50,
    **kwargs
) -> BaseEpochs:
    """Extract band power from epochs using bandpass filtering.

    This function performs bandpass filtering on epochs, applies power
    transformation (absolute value or squaring), and optionally smooths
    the result using Gaussian filtering.

    Args:
        epochs: MNE Epochs object containing the neural data.
        frange: Frequency range for bandpass filtering as [low_freq, high_freq]
            in Hz.
            Defaults to [300, 1000].
        power: Power transformation method. Either 'abs' for absolute value
            or 'square' for squaring the signal. Defaults to 'abs'.
        smooth_ms: Smoothing window size in milliseconds. Set to 0 or negative
            to skip smoothing. Defaults to 50.
        **kwargs: Additional arguments passed to epochs.filter().

    Returns:
        Processed Epochs object with band-extracted features.

    Raises:
        ValueError: If power method is not 'abs' or 'square', or if frequency
            range is invalid.

    Note:
        Automatically adjusts frequency range if upper bound at Nyquist limit
        to ensure compatibility with MNE filtering.
    """
    # Validate inputs
    if power not in ['abs', 'square']:
        raise ValueError(f"power must be 'abs' or 'square', got '{power}'")

    validate_freqs(frange)

    # Handle Nyquist edge case
    if np.allclose(epochs.info['sfreq'], 2 * frange[1]):
        # edge of band is at limit of Nyquist, subtract small amount for
        # compatibility with mne.filter
        frange = frange.copy()
        frange[1] -= 0.01

    # Copy and bandpass
    epochs = epochs.copy().filter(
        frange[0], frange[1],
        **kwargs
    )

    # Rectify
    if power == 'abs':
        epochs.apply_function(np.abs, picks='all')
    elif power == 'square':
        epochs.apply_function(lambda x: x**2, picks='all')

    # Skip smoothing if smooth_ms is 0 or negative
    if smooth_ms <= 0:
        return epochs

    # Convert smoothing from ms → samples
    sfreq = epochs.info['sfreq']
    sigma = (smooth_ms / 1000) * sfreq  # in samples

    # Apply Gaussian smoothing along time axis
    epochs.apply_function(
        lambda x: gaussian_filter1d(x, sigma=sigma, axis=-1),
        picks='all'
    )

    return epochs


def extract_band_hilbert(
    epochs: BaseEpochs,
    frange: List[float],
) -> BaseEpochs:
    """Extract band envelope using Hilbert transform.

    This function extracts the envelope of neural data in a specified frequency
    band using the Hilbert transform method. It provides an alternative to the
    bandpass filtering approach for envelope detection.

    Args:
        epochs: MNE Epochs object containing the neural data.
        frange: Frequency range for envelope extraction as [low_freq,
            high_freq] in Hz.

    Returns:
        Epochs object with Hilbert-extracted envelope features.

    Raises:
        ValueError: If frequency range is invalid.

    Note:
        Uses MNE's time-frequency analysis methods for Hilbert transform.
        The copy=False parameter modifies epochs in-place for efficiency.
    """
    # Validate inputs
    validate_freqs(frange)

    # Handle Nyquist edge case
    if epochs.info['sfreq'] == 2 * frange[1]:
        frange = frange.copy()
        frange[1] -= 0.01

    # Extract band envelope using gamma.extract
    gamma.extract(epochs, passband=frange, copy=False, n_jobs=-1)
    return epochs
