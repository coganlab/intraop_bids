import pandas as pd
from pathlib import Path
from bids import BIDSLayout
from mne_bids import BIDSPath
from ieeg.io import bidspath_from_layout

# SUBJ_LIST = ['S14', 'S22', 'S23', 'S26', 'S33', 'S39', 'S62']
SUBJ_LIST = ['S58']
COL_ORDER = ['onset', 'duration', 'value', 'trial_type', 'sample', 'subject', 'trial']
BIDS_ROOT = Path(r"~\Box\CoganLab\BIDS_1.0_Phoneme_Sequence_uECoG_share\BIDS\derivatives\phonemeLevel").expanduser()
# BIDS_ROOT = Path(r"~\Box\CoganLab\BIDS_1.0_Phoneme_Sequence_uECoG_share\BIDS").expanduser()
TASK = 'phoneme'


def load_events_tsv(bids_root, subject, task):
    """Load events.tsv from BIDS dataset."""
    # bids_layout = BIDSLayout(
    #     root=bids_root,
    #     derivatives=True,
    # )

    # bids_path = bidspath_from_layout(
    #     bids_layout,
    #     subject=subject,
    #     task=task,
    #     desc='phonemeLevel',
    #     extension='.edf',
    # )
    bids_path = BIDSPath(
        root=bids_root,
        subject=subject,
        task=task,
        acquisition='01',
        run='01',
        datatype='ieeg',
        suffix='ieeg',
        description='phonemeLevel',
        extension='.edf',
        check=False
    )

    events_tsv_path = bids_path.copy().update(suffix='events',
                                              extension='.tsv')
    events_df = pd.read_csv(events_tsv_path, sep='\t')
    return events_df, events_tsv_path


def reorder_events_columns(events_df, desired_order):
    """Reorder columns of events DataFrame to match desired order."""
    current_columns = events_df.columns.tolist()
    reordered_columns = [col for col in desired_order if col in current_columns]
    reordered_df = events_df[reordered_columns]
    return reordered_df


def save_events_tsv(events_df, events_tsv_path):
    """Save events DataFrame to events.tsv file."""
    events_df.to_csv(events_tsv_path, sep='\t', index=False)
    print(f'Saved reordered events to {events_tsv_path}')


def reorder_bids_events_columns(bids_root, subject, task, desired_order):
    """Reorder columns of events.tsv in BIDS dataset."""
    events_df, events_tsv_path = load_events_tsv(bids_root, subject, task)
    reordered_df = reorder_events_columns(events_df, desired_order)
    save_events_tsv(reordered_df, events_tsv_path)


if __name__ == "__main__":
    for subj in SUBJ_LIST:
        reorder_bids_events_columns(BIDS_ROOT, subj, TASK, COL_ORDER)