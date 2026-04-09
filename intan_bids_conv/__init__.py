"""
ECoG BIDS Conversion Tool

A Python package for converting intraoperative ECoG/iEEG data recorded in
Intan RHD format to BIDS.

Pipeline (2 user calls):
  1. ``bids_writer.main()`` -- load Intan data, run MFA, write BIDS with
     standardised word-level *and* phoneme-level events.
  2. ``update_events_from_txt.main()`` -- after manual correction, update
     both word-level and phoneme-level events in a single call.
"""

__version__ = "0.2.0"

from .bids_writer import BIDSConverter
from .update_events_from_txt import update_events_from_txt
