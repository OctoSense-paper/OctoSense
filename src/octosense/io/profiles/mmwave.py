"""mmWave board and geometry profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from octosense.io.semantics.loader import load_reader_definition_bundle

_C = 299_792_458.0
_CONFIG = load_reader_definition_bundle("mmwave", "ti_dca1000").device_config
_PRESETS = _CONFIG["presets"]
_DEFAULT_RADAR_CONFIG = _CONFIG["default_radar_config"]


def build_linear_antenna_positions(
    center_freq_hz: float,
    count: int,
    speed_of_light_mps: float = _C,
) -> tuple[tuple[float, float, float], ...]:
    spacing = speed_of_light_mps / center_freq_hz / 2.0
    return tuple((idx * spacing, 0.0, 0.0) for idx in range(count))


@dataclass(frozen=True)
class RadarConfig:
    num_tx: int
    num_rx: int
    num_adc_samples: int
    num_chirps_per_frame: int
    num_frames: int = int(_DEFAULT_RADAR_CONFIG["num_frames"])
    sample_rate: float | None = None
    center_freq: float | None = None
    bandwidth: float | None = None
    chirp_period: float | None = None
    antenna_positions: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.sample_rate is None:
            object.__setattr__(self, "sample_rate", float(_DEFAULT_RADAR_CONFIG["sample_rate"]))
        if self.center_freq is None:
            object.__setattr__(self, "center_freq", float(_DEFAULT_RADAR_CONFIG["center_freq"]))
        if self.bandwidth is None:
            object.__setattr__(self, "bandwidth", float(_DEFAULT_RADAR_CONFIG["bandwidth"]))
        if self.chirp_period is None:
            object.__setattr__(self, "chirp_period", float(_DEFAULT_RADAR_CONFIG["chirp_period"]))
        if self.num_tx <= 0:
            raise ValueError(f"num_tx must be > 0, got {self.num_tx}")
        if self.num_rx <= 0:
            raise ValueError(f"num_rx must be > 0, got {self.num_rx}")
        if self.num_adc_samples <= 0:
            raise ValueError(f"num_adc_samples must be > 0, got {self.num_adc_samples}")
        if self.num_chirps_per_frame <= 0:
            raise ValueError(
                f"num_chirps_per_frame must be > 0, got {self.num_chirps_per_frame}"
            )
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.center_freq <= 0:
            raise ValueError(f"center_freq must be > 0, got {self.center_freq}")
        if self.bandwidth <= 0:
            raise ValueError(f"bandwidth must be > 0, got {self.bandwidth}")
        if self.chirp_period <= 0:
            raise ValueError(f"chirp_period must be > 0, got {self.chirp_period}")
        if self.antenna_positions is not None and len(self.antenna_positions) != self.num_virtual_antennas:
            raise ValueError(
                f"antenna_positions length ({len(self.antenna_positions)}) must equal "
                f"num_virtual_antennas ({self.num_virtual_antennas} = {self.num_tx} * {self.num_rx})"
            )

    @property
    def num_virtual_antennas(self) -> int:
        return self.num_tx * self.num_rx

    @property
    def wavelength(self) -> float:
        return _C / self.center_freq

    @property
    def range_resolution(self) -> float:
        return _C / (2.0 * self.bandwidth)

    @property
    def max_range(self) -> float:
        return self.range_resolution * self.num_adc_samples / 2.0

    @property
    def doppler_resolution(self) -> float:
        return self.wavelength / (2.0 * self.num_chirps_per_frame * self.chirp_period)

    @property
    def max_velocity(self) -> float:
        return self.doppler_resolution * self.num_chirps_per_frame / 2.0

    def expected_file_size(self, num_frames: int | None = None) -> int:
        nf = num_frames if num_frames is not None else self.num_frames
        if nf <= 0:
            raise ValueError("num_frames must be > 0 to compute expected file size")
        return nf * self.num_chirps_per_frame * self.num_tx * self.num_rx * self.num_adc_samples * 4

    def to_dict(self) -> dict:
        payload = asdict(self)
        if payload["antenna_positions"] is not None:
            payload["antenna_positions"] = [list(item) for item in payload["antenna_positions"]]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "RadarConfig":
        data = dict(payload)
        if data.get("antenna_positions") is not None:
            data["antenna_positions"] = tuple(tuple(item) for item in data["antenna_positions"])
        return cls(**data)

    @classmethod
    def from_mmwave_studio_dict(cls, payload: dict) -> "RadarConfig":
        try:
            device = payload["mmWaveDevices"][0]
            rf_config = device["rfConfig"]
            channel_cfg = rf_config["rlChanCfg_t"]
            profile_cfg = rf_config["rlProfiles"][0]["rlProfileCfg_t"]
            chirp_cfg = rf_config["rlChirps"][0]["rlChirpCfg_t"]
            frame_cfg = rf_config["rlFrameCfg_t"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Invalid TI mmWave Studio config payload") from exc
        tx_enable = int(chirp_cfg["txEnable"], 16)
        rx_enable = int(channel_cfg["rxChannelEn"], 16)
        ramp_end_usec = float(profile_cfg["rampEndTime_usec"])
        slope_mhz_per_usec = float(profile_cfg["freqSlopeConst_MHz_usec"])
        return cls(
            num_tx=bin(tx_enable).count("1"),
            num_rx=bin(rx_enable).count("1"),
            num_adc_samples=int(profile_cfg["numAdcSamples"]),
            num_chirps_per_frame=int(frame_cfg["numLoops"]),
            num_frames=int(frame_cfg["numFrames"]),
            sample_rate=float(profile_cfg["digOutSampleRate"]) * 1000.0,
            center_freq=float(profile_cfg["startFreqConst_GHz"]) * 1e9,
            bandwidth=slope_mhz_per_usec * ramp_end_usec * 1e6,
            chirp_period=(float(profile_cfg["idleTimeConst_usec"]) + ramp_end_usec) * 1e-6,
        )

    @classmethod
    def from_mmwave_studio_json(cls, path: str | Path) -> "RadarConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(
                "TI mmWave Studio JSON must decode to an object, "
                f"got {type(payload)!r}"
            )
        return cls.from_mmwave_studio_dict(payload)

    @classmethod
    def _from_named_preset(cls, preset_name: str) -> "RadarConfig":
        payload = dict(_PRESETS[preset_name])
        payload["antenna_positions"] = build_linear_antenna_positions(
            center_freq_hz=float(payload.pop("virtual_array_center_freq_hz")),
            count=int(payload.pop("virtual_array_count")),
        )
        return cls.from_dict(payload)

    @classmethod
    def IWR1443(cls) -> "RadarConfig":
        return cls._from_named_preset("IWR1443")

    @classmethod
    def AWR1843(cls) -> "RadarConfig":
        return cls._from_named_preset("AWR1843")

    @classmethod
    def IWR6843(cls) -> "RadarConfig":
        return cls._from_named_preset("IWR6843")


__all__ = ["RadarConfig", "build_linear_antenna_positions"]
