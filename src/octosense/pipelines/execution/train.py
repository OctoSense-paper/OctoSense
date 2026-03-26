"""Shared telemetry and metric helpers for contract-native execution runs."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

PROFILE_STAGE_ORDER = (
    "loader_wait_sec",
    "cpu_prefix_sec",
    "h2d_sec",
    "gpu_preprocess_sec",
    "forward_sec",
    "loss_sec",
    "backward_sec",
    "optim_sec",
    "batch_total_sec",
)


def format_stage_profile(profile: dict[str, float]) -> str:
    """Render a compact stage-profile line in a stable field order."""
    ordered_keys = [key for key in PROFILE_STAGE_ORDER if key in profile]
    ordered_keys.extend(key for key in profile if key not in PROFILE_STAGE_ORDER)
    return ", ".join(f"{key.removesuffix('_sec')}={float(profile[key]):.4f}s" for key in ordered_keys)


@dataclass
class MetricTrace:
    """Accumulate runtime timing and train/eval trace events for one run."""

    first_batch_sec: float | None = None
    epoch_durations_sec: list[float] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    first_batch_profiles_sec: dict[str, dict[str, float]] = field(default_factory=dict)

    def maybe_record_first_batch(self, duration_sec: float) -> None:
        if self.first_batch_sec is None:
            self.first_batch_sec = float(duration_sec)

    def record_first_batch_profile(
        self,
        *,
        split: str,
        profile: dict[str, float],
    ) -> None:
        if split in self.first_batch_profiles_sec:
            return
        normalized = {str(key): float(value) for key, value in profile.items()}
        self.first_batch_profiles_sec[split] = normalized
        self.events.append(
            {
                "event": "first_batch_profile",
                "split": str(split),
                **normalized,
            }
        )

    def record_epoch_end(
        self,
        *,
        epoch: int,
        train_loss: float,
        epoch_duration_sec: float,
        val_loss: float | None = None,
        metric_name: str | None = None,
        metric_value: float | None = None,
    ) -> None:
        duration = float(epoch_duration_sec)
        payload: dict[str, object] = {
            "event": "epoch_end",
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "epoch_duration_sec": duration,
        }
        if val_loss is not None:
            payload["val_loss"] = float(val_loss)
        if metric_name is not None and metric_value is not None:
            payload[str(metric_name)] = float(metric_value)
        self.epoch_durations_sec.append(duration)
        self.events.append(payload)

    def record_classification_eval(
        self,
        *,
        split: str,
        accuracy: float,
        correct: int,
        total: int,
        loss: float | None = None,
        epoch: int | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "event": "eval",
            "split": split,
            "accuracy": float(accuracy),
            "correct": int(correct),
            "total": int(total),
        }
        if loss is not None:
            payload["loss"] = float(loss)
        if epoch is not None:
            payload["epoch"] = int(epoch)
        self.events.append(payload)

    def record_structured_eval(
        self,
        *,
        split: str,
        loss: float,
        metric_name: str,
        metric_value: float,
        batches: int,
    ) -> None:
        self.events.append(
            {
                "event": "eval",
                "split": split,
                "loss": float(loss),
                str(metric_name): float(metric_value),
                "batches": int(batches),
            }
        )

    def build_timing_payload(
        self,
        *,
        duration_sec: float,
        peak_memory_mb_value: float,
        train_samples: int,
        epochs: int,
    ) -> dict[str, object]:
        total_train_samples = max(0, int(train_samples)) * max(0, int(epochs))
        duration = float(duration_sec)
        mean_epoch_sec = (
            float(sum(self.epoch_durations_sec) / len(self.epoch_durations_sec))
            if self.epoch_durations_sec
            else 0.0
        )
        throughput = float(total_train_samples / duration) if duration > 0.0 else 0.0
        return {
            "first_batch_sec": float(self.first_batch_sec or 0.0),
            "epoch_durations_sec": [float(value) for value in self.epoch_durations_sec],
            "mean_epoch_sec": mean_epoch_sec,
            "duration_sec": duration,
            "peak_memory_mb": float(peak_memory_mb_value),
            "train_samples_processed": float(total_train_samples),
            "train_throughput_samples_per_sec": throughput,
            "epochs_completed": float(len(self.epoch_durations_sec)),
            "first_batch_profiles_sec": {
                str(split): {str(key): float(value) for key, value in profile.items()}
                for split, profile in self.first_batch_profiles_sec.items()
            },
            "first_train_batch_profile_sec": {
                str(key): float(value)
                for key, value in self.first_batch_profiles_sec.get("train", {}).items()
            },
        }
def peak_memory_mb(device: torch.device) -> float:
    """Return the current device peak memory footprint in MB."""
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    if device.type == "mps":
        current_allocated = getattr(torch.mps, "current_allocated_memory", None)
        if callable(current_allocated):
            return float(current_allocated() / (1024**2))
    return 0.0


def synchronize_device(device: torch.device) -> None:
    """Block until queued work on the selected backend completes."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "mps":
        synchronize = getattr(torch.mps, "synchronize", None)
        if callable(synchronize):
            synchronize()


__all__ = [
    "MetricTrace",
    "PROFILE_STAGE_ORDER",
    "format_stage_profile",
    "peak_memory_mb",
    "synchronize_device",
]
