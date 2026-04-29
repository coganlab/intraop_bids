"""Phoneme-level helpers for ARPAbet event remapping."""

import re

import numpy as np


def remove_arpabet_stress(phoneme: str) -> str:
    """Remove stress markers from ARPAbet phonemes.

    E.g., 'AH0' -> 'AH', 'IY1' -> 'IY'
    """
    return re.sub(r'\d', '', phoneme)


def remap_phoneme_events(epochs):
    """Collapse stress variants into canonical phoneme event IDs.

    Strips stress digits and the annotation path prefix from each event
    description, then reassigns event codes so that identical phonemes
    share the same integer code (alphabetically ordered starting at 1).

    Modifies ``epochs.event_id`` and ``epochs.events`` in place and
    returns the epochs object.
    """
    old_id = epochs.event_id
    ev2phon = {v: remove_arpabet_stress(k.split('/')[-1])
               for k, v in old_id.items()}

    unique_phons = sorted(set(ev2phon.values()))
    new_id = {phon: i + 1 for i, phon in enumerate(unique_phons)}

    new_events = epochs.events.copy()
    for i, ev in enumerate(new_events):
        new_events[i, 2] = new_id[ev2phon[ev[2]]]

    epochs.event_id = new_id
    epochs.events = new_events
    return epochs
