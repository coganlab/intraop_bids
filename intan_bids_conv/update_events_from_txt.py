"""
Update events.tsv files in a BIDS dataset from manually corrected .txt files.

Updates both word-level events (in the main BIDS directory) and phoneme-level
events (in ``derivatives/phonemeLevel/``) in a single call.

Steps:
1) Load subject raw from BIDS to obtain the sampling frequency.
2) Update word-level events.tsv from ``stim_txt`` / ``resp_txt``.
3) Update phoneme-level events.tsv from ``stim_phon_txt`` / ``resp_phon_txt``.
4) Write ``_events.json`` sidecars for both levels.

Run via Hydra, e.g.::

    python update_events_from_txt.py subject=S85 task=lexical
"""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from bids import BIDSLayout
from ieeg.io import bidspath_from_layout, raw_from_layout
from mne_bids import BIDSPath
from omegaconf import DictConfig, OmegaConf

try:
    from bids_writer import (
        EVENTS_COL_ORDER,
        EVENTS_JSON_METADATA,
        NPHONS,
        PHONEME_EVENTS_JSON_METADATA,
        phon_txt2df,
        remove_orphan_stimuli,
    )
except ImportError:
    from .bids_writer import (
        EVENTS_COL_ORDER,
        EVENTS_JSON_METADATA,
        NPHONS,
        PHONEME_EVENTS_JSON_METADATA,
        phon_txt2df,
        remove_orphan_stimuli,
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_DATA_DIR = Path(
    "~/Box/CoganLab/Data/Micro/BIDS_processing"
).expanduser()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_events_from_txt(
    bids_root,
    subject,
    task,
    stim_txt=None,
    resp_txt=None,
    stim_phon_txt=None,
    resp_phon_txt=None,
    response_window_sec=3.0,
):
    """Update word- and phoneme-level events.tsv from corrected .txt files.

    Parameters
    ----------
    bids_root : str or Path
        Root directory of the BIDS dataset.
    subject : str
        Subject identifier (without ``sub-`` prefix).
    task : str
        BIDS task label.
    stim_txt, resp_txt : str or Path, optional
        Paths to manually corrected word-level stimulus/response .txt files.
    stim_phon_txt, resp_phon_txt : str or Path, optional
        Paths to manually corrected phoneme-level stimulus/response .txt files.
    response_window_sec : float, optional
        Window in seconds to consider for response events.
        Default is 3.0 seconds.
        This is used to remove orphan stimuli from the events.tsv files.
        If the response window is too short, some stimuli may be removed
        unnecessarily.
        If the response window is too long, some responses may be included
        that are not part of the stimulus.
    """
    has_word = stim_txt is not None or resp_txt is not None
    has_phon = stim_phon_txt is not None or resp_phon_txt is not None

    if not has_word and not has_phon:
        logger.info("No .txt files provided for updating events. Exiting.")
        return

    bids_layout = BIDSLayout(root=bids_root, derivatives=False)

    raw = raw_from_layout(
        bids_layout,
        subject=subject,
        task=task,
        extension=".edf",
        preload=False,
    )
    fs = raw.info["sfreq"]

    bids_path = bidspath_from_layout(
        bids_layout,
        subject=subject,
        task=task,
        extension=".edf",
    )

    if has_word:
        _update_word_events(
            bids_path, subject, fs, stim_txt, resp_txt, response_window_sec
        )

    if has_phon:
        _update_phoneme_events(
            bids_root,
            subject,
            task,
            fs,
            stim_phon_txt,
            resp_phon_txt,
            response_window_sec,
        )


# ---------------------------------------------------------------------------
# Word-level helpers
# ---------------------------------------------------------------------------


def _update_word_events(
    bids_path, subject, fs, stim_txt, resp_txt, response_window_sec
):
    """Replace word-level events.tsv with data from corrected txt files."""
    events_tsv_path = bids_path.copy().update(
        suffix="events",
        extension=".tsv",
    )
    events_df = pd.read_csv(events_tsv_path.fpath, sep="\t")
    stim_events_df = events_df[events_df["trial_type"] == "stimulus"]
    resp_events_df = events_df[events_df["trial_type"] == "response"]

    if stim_txt:
        stim_events_df = _word_txt2df(stim_txt, "stimulus", subject, fs)
    if resp_txt:
        resp_events_df = _word_txt2df(resp_txt, "response", subject, fs)

    updated_df = (
        pd.concat([stim_events_df, resp_events_df], ignore_index=True)
        .sort_values("onset")
        .reset_index(drop=True)
    )

    updated_df = remove_orphan_stimuli(updated_df, response_window_sec)

    # Check that stimulus and response counts match after removing orphans.
    stim_events_updated = updated_df[updated_df["trial_type"] == "stimulus"]
    resp_events_updated = updated_df[updated_df["trial_type"] == "response"]
    # print(len(stim_events_updated), len(resp_events_updated))
    if len(stim_events_updated) != len(resp_events_updated):
        logger.error(
            "Stimulus and response counts differ after removing orphan stimuli"
        )
        return

    updated_df["trial"] = np.ceil(
        np.arange(1, len(updated_df) + 1) / 2
    ).astype(int)

    updated_df = updated_df[EVENTS_COL_ORDER]
    updated_df.to_csv(events_tsv_path.fpath, sep="\t", index=False)

    _write_json(
        events_tsv_path.fpath.with_suffix(".json"),
        EVENTS_JSON_METADATA,
    )
    logger.info("Updated word-level events.tsv at %s", events_tsv_path.fpath)


def _word_txt2df(txt_file, trial_type, subject, fs):
    """Convert a word-level MFA .txt file to a BIDS events DataFrame."""
    txt_file = _resolve_txt_path(txt_file, subject)
    event_df = pd.read_csv(
        txt_file,
        sep="\t",
        names=["onset", "offset", "value"],
    )
    event_df["duration"] = event_df["offset"] - event_df["onset"]
    event_df["trial_type"] = trial_type
    event_df["sample"] = (event_df["onset"] * fs).astype(int)
    event_df["subject"] = subject
    event_df["trial"] = range(1, len(event_df) + 1)
    return event_df[EVENTS_COL_ORDER]


# ---------------------------------------------------------------------------
# Phoneme-level helpers
# ---------------------------------------------------------------------------


def _update_phoneme_events(
    bids_root,
    subject,
    task,
    fs,
    stim_phon_txt,
    resp_phon_txt,
    response_window_sec,
):
    """Replace phoneme-level events.tsv in derivatives/phonemeLevel/."""
    bids_root = Path(bids_root)
    n_phons = NPHONS.get(task, 5)

    if stim_phon_txt is None or resp_phon_txt is None:
        logger.warning(
            "Both stim_phon_txt and resp_phon_txt are required for "
            "phoneme-level update. Skipping.",
        )
        return

    stim_phon_txt = _resolve_txt_path(stim_phon_txt, subject)
    resp_phon_txt = _resolve_txt_path(resp_phon_txt, subject)

    stim_df = phon_txt2df(
        stim_phon_txt,
        "stimulus",
        subject,
        n_phons,
        fs,
        task,
    )
    resp_df = phon_txt2df(
        resp_phon_txt,
        "response",
        subject,
        n_phons,
        fs,
        task,
    )

    events_df = (
        pd.concat([stim_df, resp_df])
        .sort_values("onset")
        .reset_index(drop=True)
    )
    events_df = remove_orphan_stimuli(events_df, response_window_sec)

    # Check that stimulus and response counts match after removing orphans.
    stim_events_updated = events_df[events_df["trial_type"] == "stimulus"]
    resp_events_updated = events_df[events_df["trial_type"] == "response"]
    if len(stim_events_updated) != len(resp_events_updated):
        logger.error(
            "Stimulus and response counts differ after removing orphan stimuli"
        )
        return

    events_df["trial"] = np.ceil(
        np.arange(1, len(events_df) + 1) / (n_phons * 2)
    ).astype(int)

    events_path = BIDSPath(
        root=bids_root / "derivatives" / "phonemeLevel",
        subject=subject,
        task=task,
        datatype="ieeg",
        suffix="events",
        description="phonemeLevel",
        extension=".tsv",
        check=False,
    )
    events_path.mkdir(exist_ok=True)
    events_df.to_csv(events_path.fpath, sep="\t", index=False)

    _write_json(
        events_path.fpath.with_suffix(".json"),
        PHONEME_EVENTS_JSON_METADATA,
    )
    logger.info(
        "Updated phoneme-level events.tsv at %s",
        events_path.fpath,
    )


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _resolve_txt_path(txt_path, subject):
    """Resolve a txt file path, falling back to DEFAULT_DATA_DIR if needed."""
    txt_path = Path(txt_path)
    if txt_path.exists():
        return txt_path
    fallback_adj = DEFAULT_DATA_DIR / f"sub-{subject}" / txt_path
    if "adj" in txt_path.name:
        fallback_orig = Path(str(fallback_adj).replace("mfa_adj_", "mfa_"))
    else:
        fallback_orig = Path("\0")
    if fallback_adj.exists():
        return fallback_adj
    elif fallback_orig.exists():
        return fallback_orig
    raise FileNotFoundError(
        f"Text file not found at {txt_path}, {fallback_adj}, or "
        f"{fallback_orig}"
    )


def _write_json(json_path, metadata):
    """Write a JSON sidecar file."""
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _optional_path(value: str | None) -> Path | None:
    """Return an expanded Path, or None if *value* is null."""
    if value is None:
        return None
    return Path(value).expanduser()


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="update_events_from_txt",
)
def main(cfg: DictConfig) -> None:
    """Update BIDS events.tsv files from corrected MFA .txt files."""
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Missing configuration keys: {missing_keys}")

    logger.info(f"Running with configuration:\n{OmegaConf.to_yaml(cfg)}")

    subject = cfg.subject
    task = cfg.task

    if task == "lexical":
        bids_root = (
            Path.home()
            / "Box"
            / "CoganLab"
            / "BIDS_1.0_Lexical_\u00b5ECoG"
            / "BIDS"
        )
    elif task == "phoneme":
        bids_root = (
            Path.home()
            / "Box"
            / "CoganLab"
            / "BIDS_1.0_Phoneme_Sequence_uECoG"
            / "BIDS"
        )
    else:
        raise ValueError(f"Invalid task: {task}")

    update_events_from_txt(
        bids_root=bids_root,
        subject=subject,
        task=task,
        stim_txt=_optional_path(cfg.stim_txt),
        resp_txt=_optional_path(cfg.resp_txt),
        stim_phon_txt=_optional_path(cfg.stim_phon_txt),
        resp_phon_txt=_optional_path(cfg.resp_phon_txt),
        response_window_sec=cfg.response_window_sec,
    )


if __name__ == "__main__":
    main()
