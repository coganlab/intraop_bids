"""Utilities for parsing subject identifier strings."""

from __future__ import annotations

import re

_BASE_SUBJECT_PATTERN = re.compile(r"^[A-Za-z]+\d+")


def base_subject_id(subject: str) -> str:
    """Return the core subject ID used for RHD files and recon data.

    Strips an optional ``sub-`` prefix, then returns the leading letter
    sequence plus the immediately following digit sequence. Any trailing run
    suffix (e.g. ``a`` in ``S84a``) is removed.

    Examples
    --------
    >>> base_subject_id('S84')
    'S84'
    >>> base_subject_id('S84a')
    'S84'
    >>> base_subject_id('S8412')
    'S8412'
    >>> base_subject_id('SY845')
    'SY845'
    >>> base_subject_id('sub-S84b')
    'S84'
    """
    label = subject.strip()
    if label.lower().startswith("sub-"):
        label = label[4:]

    match = _BASE_SUBJECT_PATTERN.match(label)
    if match is None:
        raise ValueError(
            f"Cannot extract base subject ID from {subject!r}. "
            "Expected format: letters followed by digits, with an optional "
            "run suffix (e.g. S84, S84a).",
        )
    return match.group(0)
