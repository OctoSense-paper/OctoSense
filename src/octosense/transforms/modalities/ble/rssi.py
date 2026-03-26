"""BLE-specific transforms.

Provides signal processing transforms for BLE data:
- RSSINormalize: Normalize RSSI dBm values to [0, 1] range
- ChannelHopping: Reorder channels by BLE frequency-hopping sequence
"""

import time

import torch

from octosense.core.contracts import AxisContract
from octosense.core.errors import DimensionError
from octosense.io.semantics.metadata import TransformRecord
from octosense.io.semantics.schema import AxisSchema
from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


# Standard BLE RSSI range (dBm)
_RSSI_MIN_DBM = -100.0
_RSSI_MAX_DBM = 0.0

# BLE frequency-hopping sequence (37 data channels in hop order)
# Based on BLE specification: channels are remapped via hop increment
# Default hop sequence for 37 data channels (channel indices 0-36)
_BLE_HOP_SEQUENCE = [
    0, 6, 12, 18, 24, 30, 36, 5, 11, 17,
    23, 29, 35, 4, 10, 16, 22, 28, 34, 3,
    9, 15, 21, 27, 33, 2, 8, 14, 20, 26,
    32, 1, 7, 13, 19, 25, 31,
]


@registered_operator(required_axes=["channel"], required_meta=[])
class RSSINormalize(BaseTransform):
    """Normalize BLE RSSI values from dBm to [0, 1] range.

    Applies min-max normalization using configurable RSSI bounds:
        rssi_norm = (rssi - rssi_min) / (rssi_max - rssi_min)

    Values are clamped to [0, 1] after normalization.

    Args:
        rssi_min: Minimum RSSI in dBm (mapped to 0). Default: -100.
        rssi_max: Maximum RSSI in dBm (mapped to 1). Default: 0.
    """

    def __init__(
        self,
        rssi_min: float = _RSSI_MIN_DBM,
        rssi_max: float = _RSSI_MAX_DBM,
    ) -> None:
        super().__init__()
        self.rssi_min = rssi_min
        self.rssi_max = rssi_max

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=["channel"],
            dtype_constraint="real",
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=["channel"],
            dtype_constraint="real",
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        data = x.as_tensor()

        # Normalize: (rssi - min) / (max - min), then clamp to [0, 1]
        scale = self.rssi_max - self.rssi_min
        normalized = (data - self.rssi_min) / scale
        normalized = normalized.clamp(0.0, 1.0)

        track_meta = is_tracking_meta()
        new_metadata = x.metadata.copy() if track_meta else x.metadata
        if track_meta:
            new_metadata.transforms.append(
                TransformRecord(
                    name="RSSINormalize",
                    params={"rssi_min": self.rssi_min, "rssi_max": self.rssi_max},
                    timestamp=time.time(),
                )
            )

        return RadioTensor(
            data=normalized,
            axis_schema=x.axis_schema,
            metadata=new_metadata,
        )


@registered_operator(required_axes=[], required_meta=[])
class ChannelHopping(BaseTransform):
    """Reorder BLE channels by frequency-hopping sequence.

    BLE uses adaptive frequency hopping (AFH) across 37 data channels.
    This transform reorders the channel axis from channel-index order
    to the standard BLE hop sequence order, which can reveal temporal
    patterns masked by the hopping scheme.

    Args:
        channel_axis: Name of the channel axis. Default: "channel".
        hop_sequence: Custom hop sequence (list of channel indices).
            If None, uses the standard BLE hop sequence.
    """

    def __init__(
        self,
        channel_axis: str = "channel",
        hop_sequence: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.channel_axis = channel_axis
        self.hop_sequence = hop_sequence or _BLE_HOP_SEQUENCE

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.channel_axis],
            dtype_constraint="real",
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.channel_axis],
            dtype_constraint="real",
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        data = x.as_tensor()

        if not x.axis_schema.has_axis(self.channel_axis):
            raise DimensionError(
                f"ChannelHopping requires axis '{self.channel_axis}'.",
                available_axes=list(x.axis_schema.axes),
                suggestion=x.axis_schema.suggest_axis_name(self.channel_axis),
            )

        ch_dim = x.axis_schema.index(self.channel_axis)
        n_channels = data.shape[ch_dim]

        # Filter hop sequence to only include valid channel indices
        valid_hops = [ch for ch in self.hop_sequence if ch < n_channels]

        # Add any channels not in the hop sequence at the end
        hop_set = set(valid_hops)
        remaining = [ch for ch in range(n_channels) if ch not in hop_set]
        reorder_indices = valid_hops + remaining

        # Reorder along channel dimension
        idx = torch.tensor(reorder_indices, dtype=torch.long, device=data.device)
        reordered = data.index_select(ch_dim, idx)

        # Build new axis metadata noting the reorder
        new_axes = list(x.axis_schema.axes)
        new_metadata_dict = dict(x.axis_schema.axis_metadata)
        new_schema = AxisSchema(
            axes=tuple(new_axes),
            axis_metadata=new_metadata_dict,
        )

        track_meta = is_tracking_meta()
        new_metadata = x.metadata.copy() if track_meta else x.metadata
        if track_meta:
            new_metadata.transforms.append(
                TransformRecord(
                    name="ChannelHopping",
                    params={
                        "channel_axis": self.channel_axis,
                        "hop_sequence_len": len(reorder_indices),
                    },
                    timestamp=time.time(),
                )
            )

        return RadioTensor(
            data=reordered,
            axis_schema=new_schema,
            metadata=new_metadata,
        )
