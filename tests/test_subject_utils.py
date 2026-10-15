"""Tests for subject_utils.base_subject_id."""

import pytest

from intan_bids_conv.subject_utils import base_subject_id


@pytest.mark.parametrize(
    ('subject', 'expected'),
    [
        ('S84', 'S84'),
        ('S84a', 'S84'),
        ('S84b', 'S84'),
        ('S8412', 'S8412'),
        ('SY845', 'SY845'),
        ('sub-S84a', 'S84'),
        ('sub-S84', 'S84'),
    ],
)
def test_base_subject_id(subject, expected):
    assert base_subject_id(subject) == expected


def test_base_subject_id_invalid():
    with pytest.raises(ValueError, match='Cannot extract base subject ID'):
        base_subject_id('invalid')
