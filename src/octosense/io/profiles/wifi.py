"""WiFi channel and subcarrier profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from octosense.io.semantics.loader import load_reader_definition_bundle
from octosense.io.semantics.metadata import SignalMetadata

if TYPE_CHECKING:
    from octosense.io.tensor import RadioTensor

CHANNEL_CENTER_FREQUENCY_MHZ_2G: dict[int, float] = {
    1: 2412.0,
    2: 2417.0,
    3: 2422.0,
    4: 2427.0,
    5: 2432.0,
    6: 2437.0,
    7: 2442.0,
    8: 2447.0,
    9: 2452.0,
    10: 2457.0,
    11: 2462.0,
    12: 2467.0,
    13: 2472.0,
    14: 2484.0,
}
CHANNEL_CENTER_FREQUENCY_MHZ_5G: dict[int, float] = {
    36: 5180.0,
    40: 5200.0,
    44: 5220.0,
    48: 5240.0,
    52: 5260.0,
    56: 5280.0,
    60: 5300.0,
    64: 5320.0,
    100: 5500.0,
    104: 5520.0,
    108: 5540.0,
    112: 5560.0,
    116: 5580.0,
    120: 5600.0,
    124: 5620.0,
    128: 5640.0,
    132: 5660.0,
    136: 5680.0,
    140: 5700.0,
    144: 5720.0,
    149: 5745.0,
    153: 5765.0,
    157: 5785.0,
    161: 5805.0,
    165: 5825.0,
}
CHANNEL_CENTER_FREQUENCY_HZ: dict[int, float] = {
    **{channel: freq * 1e6 for channel, freq in CHANNEL_CENTER_FREQUENCY_MHZ_2G.items()},
    **{channel: freq * 1e6 for channel, freq in CHANNEL_CENTER_FREQUENCY_MHZ_5G.items()},
}

_IWL5300_PROFILE = load_reader_definition_bundle("wifi", "iwl5300").device_config
BASE_WIFI_SUBCARRIER_SPACING_HZ = float(_IWL5300_PROFILE["base_subcarrier_spacing_hz"])
NARROW_WIFI_SUBCARRIER_SPACING_HZ = BASE_WIFI_SUBCARRIER_SPACING_HZ / 4.0
_NARROW_SPACING_RATE_FORMATS = frozenset({"HE", "EHT"})
IWL5300_20MHZ_SUBCARRIER_INDICES = tuple(
    int(value) for value in _IWL5300_PROFILE["subcarrier_indices_20mhz"]
)
_WIFI_SUBCARRIER_UNIT_ALIASES: dict[str, str] = {
    "": "index",
    "index": "index",
    "hz": "offset_hz",
    "mhz": "offset_mhz",
    "offset_hz": "offset_hz",
    "offset_mhz": "offset_mhz",
    "freq_hz": "freq_hz",
    "freq_mhz": "freq_mhz",
}


def bandwidth_from_subcarrier_count(
    num_subc: int,
    thresholds_hz: dict[int, float],
) -> float:
    """Resolve WiFi bandwidth from a device-family subcarrier threshold table."""

    for max_subc in sorted(thresholds_hz):
        if num_subc <= max_subc:
            return float(thresholds_hz[max_subc])
    return float(thresholds_hz[max(thresholds_hz)])


def resolve_wifi_subcarrier_spacing_hz(
    *,
    rate_format: str | None = None,
    narrow_spacing: bool | None = None,
) -> float:
    """Resolve the physical WiFi subcarrier spacing from PHY mode metadata.

    WiFi CSI captures should not derive spacing from the observed subcarrier count.
    For OFDM-based WiFi families, the physical tone spacing is normally 312.5 kHz.
    HE/EHT narrow-spacing modes use 78.125 kHz instead.
    """

    if narrow_spacing is None:
        resolved_narrow = False
        if rate_format is not None:
            resolved_narrow = str(rate_format).strip().upper() in _NARROW_SPACING_RATE_FORMATS
    else:
        resolved_narrow = bool(narrow_spacing)
    return (
        NARROW_WIFI_SUBCARRIER_SPACING_HZ
        if resolved_narrow
        else BASE_WIFI_SUBCARRIER_SPACING_HZ
    )


def build_centered_wifi_subcarrier_indices(num_subc: int) -> np.ndarray:
    """Build a symmetric WiFi subcarrier-index axis around DC.

    Even-sized axes skip the DC bin so counts like 52 and 114 map to
    ``[-26..-1, 1..26]`` and ``[-57..-1, 1..57]`` respectively.
    """

    count = int(num_subc)
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    if count % 2 == 0:
        half = count // 2
        return np.concatenate(
            (
                np.arange(-half, 0, dtype=np.int64),
                np.arange(1, half + 1, dtype=np.int64),
            )
        )
    half = count // 2
    return np.arange(-half, half + 1, dtype=np.int64)


def channel_to_center_freq(channel: int) -> float:
    freq = CHANNEL_CENTER_FREQUENCY_HZ.get(channel)
    if freq is None:
        raise ValueError(f"Unknown WiFi channel number: {channel}")
    return freq


def is_5ghz_channel(channel: int) -> bool:
    return channel in CHANNEL_CENTER_FREQUENCY_MHZ_5G


def is_2ghz_channel(channel: int) -> bool:
    return channel in CHANNEL_CENTER_FREQUENCY_MHZ_2G


def normalize_wifi_subcarrier_unit(unit: str | None) -> str:
    normalized = "" if unit is None else str(unit).strip().lower()
    resolved = _WIFI_SUBCARRIER_UNIT_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unsupported WiFi subcarrier unit {unit!r}. "
            "Use one of: index, offset_hz, offset_mhz, freq_hz, freq_mhz."
        )
    return resolved


def convert_wifi_subcarrier_values(
    values: np.ndarray | list[float] | tuple[float, ...] | float,
    *,
    from_unit: str | None,
    to_unit: str | None,
    subcarrier_spacing_hz: float | None,
    center_freq_hz: float | None = None,
) -> np.ndarray:
    from_unit_normalized = normalize_wifi_subcarrier_unit(from_unit)
    to_unit_normalized = normalize_wifi_subcarrier_unit(to_unit)
    values_array = np.asarray(values, dtype=np.float64)

    def _require_spacing() -> float:
        spacing = 0.0 if subcarrier_spacing_hz is None else float(subcarrier_spacing_hz)
        if spacing <= 0.0:
            raise ValueError(
                "WiFi subcarrier conversion requires metadata.subcarrier_spacing > 0."
            )
        return spacing

    def _require_center_freq() -> float:
        if center_freq_hz is None:
            raise ValueError(
                "WiFi absolute-frequency conversion requires metadata.center_freq > 0."
            )
        return float(center_freq_hz)

    if from_unit_normalized == "index":
        index_values = values_array
    elif from_unit_normalized == "offset_hz":
        index_values = values_array / _require_spacing()
    elif from_unit_normalized == "offset_mhz":
        index_values = (values_array * 1.0e6) / _require_spacing()
    elif from_unit_normalized == "freq_hz":
        index_values = (values_array - _require_center_freq()) / _require_spacing()
    else:
        index_values = ((values_array * 1.0e6) - _require_center_freq()) / _require_spacing()

    if to_unit_normalized == "index":
        return index_values

    spacing_hz = _require_spacing()
    offset_hz = index_values * spacing_hz
    if to_unit_normalized == "offset_hz":
        return offset_hz
    if to_unit_normalized == "offset_mhz":
        return offset_hz / 1.0e6

    center_freq = _require_center_freq()
    if to_unit_normalized == "freq_hz":
        return center_freq + offset_hz
    return (center_freq + offset_hz) / 1.0e6


def get_wifi_subcarrier_indices(source: SignalMetadata | "RadioTensor") -> np.ndarray:
    metadata = source.metadata if hasattr(source, "metadata") else source
    if not isinstance(metadata, SignalMetadata):
        raise TypeError(f"Expected SignalMetadata or RadioTensor, got {type(source)!r}")
    if metadata.subcarrier_indices:
        return np.asarray(metadata.subcarrier_indices, dtype=np.float64)

    coord = metadata.get_coord("subc")
    if coord is None or coord.values is None:
        return np.asarray([], dtype=np.float64)
    return convert_wifi_subcarrier_values(
        coord.values,
        from_unit=coord.unit,
        to_unit="index",
        subcarrier_spacing_hz=metadata.subcarrier_spacing,
        center_freq_hz=metadata.center_freq if metadata.center_freq else None,
    )


def get_wifi_subcarrier_axis_values(
    source: SignalMetadata | "RadioTensor",
    *,
    unit: str | None = "index",
) -> np.ndarray:
    metadata = source.metadata if hasattr(source, "metadata") else source
    if not isinstance(metadata, SignalMetadata):
        raise TypeError(f"Expected SignalMetadata or RadioTensor, got {type(source)!r}")
    return convert_wifi_subcarrier_values(
        get_wifi_subcarrier_indices(metadata),
        from_unit="index",
        to_unit=unit,
        subcarrier_spacing_hz=metadata.subcarrier_spacing,
        center_freq_hz=metadata.center_freq if metadata.center_freq else None,
    )


def get_wifi_subcarrier_axis_label(unit: str | None = "index") -> str:
    normalized = normalize_wifi_subcarrier_unit(unit)
    if normalized == "index":
        return "Subc (index)"
    if normalized == "offset_hz":
        return "Subc offset (Hz)"
    if normalized == "offset_mhz":
        return "Subc offset (MHz)"
    if normalized == "freq_hz":
        return "Frequency (Hz)"
    return "Frequency (MHz)"


def get_wifi_subcarrier_axis_info(
    source: SignalMetadata | "RadioTensor",
    *,
    unit: str | None = "index",
) -> tuple[np.ndarray, str]:
    return get_wifi_subcarrier_axis_values(source, unit=unit), get_wifi_subcarrier_axis_label(unit)


__all__ = [
    "BASE_WIFI_SUBCARRIER_SPACING_HZ",
    "CHANNEL_CENTER_FREQUENCY_HZ",
    "CHANNEL_CENTER_FREQUENCY_MHZ_2G",
    "CHANNEL_CENTER_FREQUENCY_MHZ_5G",
    "IWL5300_20MHZ_SUBCARRIER_INDICES",
    "NARROW_WIFI_SUBCARRIER_SPACING_HZ",
    "bandwidth_from_subcarrier_count",
    "build_centered_wifi_subcarrier_indices",
    "channel_to_center_freq",
    "convert_wifi_subcarrier_values",
    "get_wifi_subcarrier_axis_info",
    "get_wifi_subcarrier_axis_label",
    "get_wifi_subcarrier_axis_values",
    "get_wifi_subcarrier_indices",
    "is_2ghz_channel",
    "is_5ghz_channel",
    "normalize_wifi_subcarrier_unit",
    "resolve_wifi_subcarrier_spacing_hz",
]
