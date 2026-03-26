"""Canonical WiFi CSI phase operators."""

import time

import torch

from octosense.core.contracts import AxisContract
from octosense.core.errors import DimensionError
from octosense.io.semantics.metadata import TransformRecord
from octosense.io.semantics.schema import AxisSchema
from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


@registered_operator(
    required_axes=[],
)
class PhaseCalibration(BaseTransform):
    """Remove linear phase slope caused by SFO/CFO."""

    def __init__(self, subc_axis: str = "subc") -> None:
        super().__init__()
        self.subc_axis = subc_axis

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.subc_axis],
            dtype_constraint="complex",
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.subc_axis],
            dtype_constraint="complex",
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        data = x.as_tensor()

        if not x.axis_schema.has_axis(self.subc_axis):
            raise DimensionError(
                f"PhaseCalibration requires axis '{self.subc_axis}'.",
                available_axes=list(x.axis_schema.axes),
                suggestion=x.axis_schema.suggest_axis_name(self.subc_axis),
            )
        subc_dim = x.axis_schema.index(self.subc_axis)
        n_subc = data.shape[subc_dim]
        data_perm = data.movedim(subc_dim, -1)

        phase_wrapped = torch.angle(data_perm)
        diff = torch.diff(phase_wrapped, dim=-1)
        diff_wrapped = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        diff_wrapped = torch.where(
            (diff_wrapped == -torch.pi) & (diff > 0),
            torch.tensor(torch.pi, dtype=diff_wrapped.dtype, device=diff_wrapped.device),
            diff_wrapped,
        )
        correction = torch.cumsum(diff_wrapped - diff, dim=-1)
        pad_shape = list(phase_wrapped.shape)
        pad_shape[-1] = 1
        zero_pad = torch.zeros(pad_shape, dtype=phase_wrapped.dtype, device=phase_wrapped.device)
        phase = phase_wrapped + torch.cat([zero_pad, correction], dim=-1)

        k = torch.arange(n_subc, dtype=phase.dtype, device=phase.device)
        k_sum = k.sum()
        k2_sum = (k * k).sum()
        n = float(n_subc)

        kp_sum = (k * phase).sum(dim=-1)
        p_sum = phase.sum(dim=-1)

        denom = n * k2_sum - k_sum * k_sum
        a = (n * kp_sum - k_sum * p_sum) / denom
        b = (p_sum - a * k_sum) / n

        linear_phase = a.unsqueeze(-1) * k + b.unsqueeze(-1)
        compensation = torch.exp(-1j * linear_phase.to(data_perm.dtype))
        calibrated = data_perm * compensation
        result = calibrated.movedim(-1, subc_dim)

        track_meta = is_tracking_meta()
        new_metadata = x.metadata.copy() if track_meta else x.metadata
        if track_meta:
            new_metadata.transforms.append(
                TransformRecord(
                    name="PhaseCalibration",
                    params={"subc_axis": self.subc_axis},
                    timestamp=time.time(),
                )
            )

        return RadioTensor(
            data=result,
            axis_schema=x.axis_schema,
            metadata=new_metadata,
        )


@registered_operator(
    required_axes=[],
)
class CSIRatio(BaseTransform):
    """Compute CSI ratio between antenna pairs to cancel common phase noise."""

    def __init__(self, rx_axis: str = "rx", rx_num: int = 1, rx_den: int = 0) -> None:
        super().__init__()
        self.rx_axis = rx_axis
        self.rx_num = rx_num
        self.rx_den = rx_den

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.rx_axis],
            dtype_constraint="complex",
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(remove_axes=[self.rx_axis])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        data = x.as_tensor()

        if not x.axis_schema.has_axis(self.rx_axis):
            raise DimensionError(
                f"CSIRatio requires axis '{self.rx_axis}'.",
                available_axes=list(x.axis_schema.axes),
                suggestion=x.axis_schema.suggest_axis_name(self.rx_axis),
            )
        rx_dim = x.axis_schema.index(self.rx_axis)
        n_rx = data.shape[rx_dim]

        if self.rx_num >= n_rx or self.rx_den >= n_rx:
            raise DimensionError(
                f"CSIRatio antenna indices out of range: rx_num={self.rx_num}, "
                f"rx_den={self.rx_den}, but {self.rx_axis} axis has size {n_rx}. "
                f"Fix: Ensure rx_num and rx_den are < {n_rx}."
            )

        num = data.select(rx_dim, self.rx_num)
        den = data.select(rx_dim, self.rx_den)

        eps = 1e-10
        eps_tensor = torch.tensor(eps, dtype=den.dtype, device=den.device)
        den_safe = torch.where(den.abs() < eps, eps_tensor, den)
        ratio = num / den_safe

        new_axes = [ax for ax in x.axis_schema.axes if ax != self.rx_axis]
        new_axis_metadata = {
            ax: meta for ax, meta in x.axis_schema.axis_metadata.items() if ax != self.rx_axis
        }
        new_schema = AxisSchema(tuple(new_axes), axis_metadata=new_axis_metadata)

        track_meta = is_tracking_meta()
        new_metadata = x.metadata.copy() if track_meta else x.metadata
        if (
            hasattr(new_metadata, "coords")
            and new_metadata.coords
            and self.rx_axis in new_metadata.coords
        ):
            new_metadata.coords = {
                key: value for key, value in new_metadata.coords.items() if key != self.rx_axis
            }

        if track_meta:
            new_metadata.transforms.append(
                TransformRecord(
                    name="CSIRatio",
                    params={
                        "rx_axis": self.rx_axis,
                        "rx_num": self.rx_num,
                        "rx_den": self.rx_den,
                    },
                    timestamp=time.time(),
                )
            )

        return RadioTensor(
            data=ratio,
            axis_schema=new_schema,
            metadata=new_metadata,
        )


@registered_operator(
    required_axes=[],
)
class PhaseUnwrap(BaseTransform):
    """Unwrap phase discontinuities along a named axis."""

    def __init__(
        self,
        axis: str = "subc",
        discont: float | None = None,
    ) -> None:
        super().__init__()
        self.axis = axis
        self.discont = discont if discont is not None else torch.pi

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.axis])

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.axis])

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        data = x.as_tensor()

        phase = torch.angle(data) if data.is_complex() else data

        if not x.axis_schema.has_axis(self.axis):
            raise DimensionError(
                f"PhaseUnwrap requires axis '{self.axis}'.",
                available_axes=list(x.axis_schema.axes),
                suggestion=x.axis_schema.suggest_axis_name(self.axis),
            )
        ax_dim = x.axis_schema.index(self.axis)

        diff = torch.diff(phase, dim=ax_dim)
        period = 2.0 * self.discont
        diff_wrapped = (diff + self.discont) % period - self.discont
        diff_wrapped = torch.where(
            (diff_wrapped == -self.discont) & (diff > 0),
            torch.tensor(self.discont, dtype=diff_wrapped.dtype, device=diff_wrapped.device),
            diff_wrapped,
        )

        correction = torch.cumsum(diff_wrapped - diff, dim=ax_dim)
        pad_shape = list(phase.shape)
        pad_shape[ax_dim] = 1
        zero_pad = torch.zeros(pad_shape, dtype=phase.dtype, device=phase.device)
        correction = torch.cat([zero_pad, correction], dim=ax_dim)
        unwrapped = phase + correction

        track_meta = is_tracking_meta()
        new_metadata = x.metadata.copy() if track_meta else x.metadata
        if track_meta:
            new_metadata.transforms.append(
                TransformRecord(
                    name="PhaseUnwrap",
                    params={"axis": self.axis, "discont": self.discont},
                    timestamp=time.time(),
                )
            )

        return RadioTensor(
            data=unwrapped,
            axis_schema=x.axis_schema,
            metadata=new_metadata,
        )


__all__ = ["CSIRatio", "PhaseCalibration", "PhaseUnwrap"]
