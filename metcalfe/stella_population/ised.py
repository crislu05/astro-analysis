"""Read BC95 Fortran unformatted .ised stellar population files."""

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from metcalfe.stella_population import config


def read_fortran_records(path):
    data = Path(path).read_bytes()
    records = []
    position = 0
    while position < len(data):
        length = struct.unpack(">i", data[position : position + 4])[0]
        record = data[position + 4 : position + 4 + length]
        closing_length = struct.unpack(
            ">i", data[position + 4 + length : position + 8 + length]
        )[0]
        if closing_length != length:
            raise ValueError(f"corrupt Fortran record at byte {position} in {path}")
        records.append(record)
        position += 8 + length
    return records


def unpack_count_and_floats(record):
    count = struct.unpack(">i", record[:4])[0]
    values = np.frombuffer(record[4 : 4 + 4 * count], dtype=">f4").astype(float)
    return values


@dataclass
class StellarPopulation:
    name: str
    ages_yr: np.ndarray
    wavelengths_aa: np.ndarray
    spectra: np.ndarray
    imf_segments: np.ndarray
    sfr_code: int
    tau_yr: float
    extras: dict = field(default_factory=dict)

    @property
    def ages_gyr(self):
        return self.ages_yr / config.YEARS_PER_GYR


def read_ised(path, name=None) -> StellarPopulation:
    records = read_fortran_records(path)

    header = records[0]
    number_of_ages = struct.unpack(">i", header[:4])[0]
    ages = np.frombuffer(header[4 : 4 + 4 * number_of_ages], dtype=">f4").astype(float)
    rest = header[4 + 4 * number_of_ages :]
    number_of_segments = struct.unpack(">i", rest[:4])[0]
    segments = (
        np.frombuffer(rest[4 : 4 + 24 * number_of_segments], dtype=">f4")
        .astype(float)
        .reshape(-1, 6)
    )
    tail = rest[4 + 24 * number_of_segments :]
    sfr_code = struct.unpack(">i", tail[12:16])[0]
    tau_yr = float(np.frombuffer(tail[16:20], dtype=">f4")[0])

    wavelengths = unpack_count_and_floats(records[1])
    spectra = np.array(
        [unpack_count_and_floats(records[2 + index]) for index in range(number_of_ages)]
    )

    extras = {}
    for offset, extra_name in enumerate(config.EXTRA_RECORD_NAMES):
        record_index = 2 + number_of_ages + offset
        if record_index < len(records) and len(records[record_index]) > 4:
            extras[extra_name] = unpack_count_and_floats(records[record_index])

    if spectra.shape != (number_of_ages, len(wavelengths)):
        raise ValueError(f"unexpected spectra shape {spectra.shape} in {path}")

    return StellarPopulation(
        name=name or Path(path).stem,
        ages_yr=ages,
        wavelengths_aa=wavelengths,
        spectra=spectra,
        imf_segments=segments,
        sfr_code=sfr_code,
        tau_yr=tau_yr,
        extras=extras,
    )