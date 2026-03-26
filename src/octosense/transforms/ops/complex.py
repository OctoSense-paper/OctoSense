"""Complex-value primitive transforms."""

import time

import numpy as np
import torch

from octosense.core.contracts import AxisContract
from octosense.core.errors import DimensionError
from octosense.io.semantics.metadata import CoordinateAxis, TransformRecord
from octosense.io.semantics.schema import AxisMetadata, AxisSchema
from octosense.io.tensor import RadioTensor
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


def _amplitude(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def _phase(x: torch.Tensor) -> torch.Tensor:
    return torch.angle(x)


@registered_operator(
    required_axes=[],
    description="Extract amplitude and phase channels from complex-valued tensors.",
)
class AmplitudePhase(BaseTransform):
    """Extract amplitude and phase from complex RadioTensor.

    Converts complex values to real-valued amplitude + phase channels.
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        phase_unwrap: bool = False,
    ) -> None:
        """Initialize AmplitudePhase transform.

        Args:
            epsilon: Small constant for numerical stability
            phase_unwrap: Whether to unwrap phase (default: False for MVP)
        """
        super().__init__()
        self.epsilon = epsilon
        self.phase_unwrap = phase_unwrap

    @property
    def input_contract(self) -> AxisContract:
        """Requires complex input dtype."""
        return AxisContract(
            required_axes=[],
            dtype_constraint="complex",
        )

    @property
    def output_contract(self) -> AxisContract:
        """Output has new feature axis with 2 channels.
        
        Note: This transform adds a 'feature' axis rather than replacing one.
        """
        return AxisContract(
            add_axes=["feature"],
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        """Extract amplitude and phase.

        Args:
            x: Input RadioTensor with complex dtype

        Returns:
            RadioTensor with real dtype and new "feature" axis
        """
        # 1. Validate input via contract system
        self._validate_input(x)
        if not x.dtype.is_complex:
            raise DimensionError(f"AmplitudePhase requires complex dtype, got {x.dtype}")

        # 2. Calculate amplitude and phase using internal tensor kernels
        data = x.as_tensor()
        amplitude = _amplitude(data)
        phase = _phase(data)

        # 3. Stack along new feature dimension
        amp_phase = torch.stack([amplitude, phase], dim=-1)

        # 4. Update axis schema
        new_axes = list(x.axis_schema.axes) + ["feature"]
        axis_metadata = dict(x.axis_schema.axis_metadata)
        axis_metadata["feature"] = AxisMetadata(
            "feature",
            None,
            "Amplitude/phase feature channels",
        )
        new_schema = AxisSchema(tuple(new_axes), axis_metadata=axis_metadata)

        # 5. Update metadata
        new_metadata = x.metadata.copy()

        # Add feature coordinate axis
        new_metadata.coords["feature"] = CoordinateAxis(
            axis_name="feature",
            values=np.array(["amplitude", "phase"]),
            unit=None,
        )

        # Append transform provenance
        new_metadata.transforms.append(
            TransformRecord(
                name="AmplitudePhase",
                params={"epsilon": self.epsilon, "phase_unwrap": self.phase_unwrap},
                timestamp=time.time(),
            )
        )

        # 6. Create output
        return RadioTensor(
            data=amp_phase,
            axis_schema=new_schema,
            metadata=new_metadata,
        )


@registered_operator(
    required_axes=[],
    description="Convert complex-valued tensors to magnitude-only real tensors.",
)
class Magnitude(BaseTransform):
    """Convert complex RadioTensor to magnitude (real-valued)."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[],
            dtype_constraint="complex",
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract()

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        if not x.dtype.is_complex:
            raise DimensionError(f"Magnitude requires complex dtype, got {x.dtype}")

        data = x.as_tensor()
        magnitude = _amplitude(data)

        new_metadata = x.metadata.copy()
        new_metadata.transforms.append(
            TransformRecord(
                name="Magnitude",
                params={"epsilon": self.epsilon},
                timestamp=time.time(),
            )
        )

        return RadioTensor(
            data=magnitude,
            axis_schema=x.axis_schema,
            metadata=new_metadata,
        )


__all__ = ["AmplitudePhase", "Magnitude"]
