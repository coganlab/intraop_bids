# Adrian Foy September 2023

"""Interacts with text, for example reading Qt style QStrings or printing
reports to console.
"""

import os
import struct


def read_qstring(fid):
    """Reads Qt style QString.

    The first 32-bit unsigned number indicates the length of the string
    (in bytes). If this number equals 0xFFFFFFFF, the string is null.

    Strings are stored as unicode.
    """
    (length,) = struct.unpack("<I", fid.read(4))
    if length == int("ffffffff", 16):
        return ""

    if length > (os.fstat(fid.fileno()).st_size - fid.tell() + 1):
        print(length)
        raise QStringError("Length too long.")

    # Convert length from bytes to 16-bit Unicode words.
    length = int(length / 2)

    data = []
    for _ in range(length):
        (c,) = struct.unpack("<H", fid.read(2))
        data.append(c)

    a = "".join([chr(c) for c in data])

    return a


def print_record_time_summary(num_amp_samples, sample_rate, data_present):
    """Prints summary of how much recorded data is present in RHD file
    to console.
    """
    record_time = num_amp_samples / sample_rate

    if data_present:
        print(
            f"File contains {record_time:0.3f} seconds of data.  "
            f"Amplifiers were sampled at {sample_rate / 1000:0.2f} kS/s."
        )
    else:
        print(
            "Header file contains no data.  "
            f"Amplifiers were sampled at {sample_rate / 1000:0.2f} kS/s."
        )


def print_progress(i, target, print_step, percent_done):
    """Prints progress of an arbitrary process based on position i / target,
    printing a line showing completion percentage for each print_step / 100.
    """
    fraction_done = 100 * (1.0 * i / target)
    if fraction_done >= percent_done:
        print(f"{percent_done}% done...")
        percent_done += print_step

    return percent_done


class QStringError(Exception):
    """Raised when reading a QString fails because it is too long."""
