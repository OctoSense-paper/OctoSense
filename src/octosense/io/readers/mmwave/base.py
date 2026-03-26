"""Base class for mmWave radar readers."""

from abc import ABC, abstractmethod
from pathlib import Path

from octosense.io.readers._base import CanonicalReader, ReaderError
from octosense.io.tensor import RadioTensor


class BaseRadarReader(CanonicalReader, ABC):
    """Abstract base for all mmWave radar readers.

    Follows the same hardware-named pattern as BaseWiFiReader.
    Each concrete reader is named after the capture hardware (e.g., TI_DCA1000Reader).
    """

    modality: str = "mmwave"
    device_family: str = "generic"
    device_name: str = "UnknownRadar"
    reader_version: str = "1.0"

    @abstractmethod
    def read_file(
        self,
        file_path: str | Path,
        config: "RadarConfig",  # noqa: F821
    ) -> RadioTensor:
        """Read radar ADC data from binary file.

        Args:
            file_path: Path to raw ADC binary file.
            config: Radar configuration with board parameters.

        Returns:
            RadioTensor containing mmWave signal semantics.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is invalid or size mismatches config.
        """

    @abstractmethod
    def validate_format(
        self,
        file_path: str | Path,
        config: "RadarConfig",  # noqa: F821
    ) -> tuple[bool, str]:
        """Validate file format and size against config.

        Returns:
            (is_valid, error_message). error_message is empty when valid.
        """


__all__ = ["BaseRadarReader", "ReaderError"]
