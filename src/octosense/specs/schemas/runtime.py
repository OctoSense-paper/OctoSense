"""Runtime schema owned by ``octosense.specs``."""

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeSpec:
    """Execution-time overrides for a benchmark run."""

    device: str = "cpu"
    batch_size: int = 32
    epochs: int = 1
    seed: int = 41
    num_workers: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "num_workers": self.num_workers,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "RuntimeSpec":
        if payload is None:
            return cls()
        return cls(
            device=str(payload.get("device", "cpu") or "cpu"),
            batch_size=int(payload.get("batch_size", 32)),
            epochs=int(payload.get("epochs", 1)),
            seed=int(payload.get("seed", 41)),
            num_workers=int(payload.get("num_workers", 0)),
        )


__all__ = ["RuntimeSpec"]
