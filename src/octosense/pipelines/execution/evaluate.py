"""Shared forward-evaluation helpers for pipeline execution actions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from octosense.models.boundary import validate_model_output


@dataclass(frozen=True)
class LossEvaluation:
    loss: float
    batches: int
    samples: int


@dataclass(frozen=True)
class ForwardEvaluation:
    loss: float | None
    batches: int
    samples: int
    predictions: torch.Tensor | None
    targets: torch.Tensor | None


def _structured_target_schema(targets: Mapping[str, torch.Tensor]) -> dict[str, list[int]]:
    return {str(key): list(value.shape[1:]) for key, value in targets.items()}


def evaluate_loader(
    *,
    model: nn.Module,
    device: torch.device,
    loader: DataLoader | None,
    target_mode: Literal["tensor", "mapping"],
    resolve_prediction_target: Callable[[Any, Any], tuple[torch.Tensor, torch.Tensor]] | None,
    compute_loss: Callable[
        [Any, Any, torch.Tensor | None, torch.Tensor | None],
        torch.Tensor,
    ]
    | None = None,
) -> ForwardEvaluation:
    """Evaluate a dataloader and optionally aggregate prediction/target tensors.

    Some public loss-only helpers operate on structured mapping outputs directly and
    therefore do not need a single prediction/target tensor pair. In that mode,
    callers may pass ``resolve_prediction_target=None`` and consume only ``loss``,
    ``batches``, and ``samples`` from the returned evaluation.

    When ``compute_loss`` is provided, it is called once per batch and must return a
    single scalar tensor for that batch. The reported ``loss`` is the arithmetic mean
    of those per-batch scalar values; any aggregation across structured output fields
    is the caller's responsibility inside ``compute_loss``.
    """
    if loader is None:
        return ForwardEvaluation(
            loss=0.0 if compute_loss is not None else None,
            batches=0,
            samples=0,
            predictions=None,
            targets=None,
        )

    was_training = bool(getattr(model, "training", False))
    model.eval()
    total_loss = 0.0
    batches = 0
    samples = 0
    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for batch_x, batch_targets in loader:
                batch_x = batch_x.to(device)
                if target_mode == "mapping":
                    if not isinstance(batch_targets, Mapping):
                        raise TypeError(
                            "Mapping evaluation helpers require mapping batch targets, got "
                            f"{type(batch_targets)!r}."
                        )
                    targets = {key: value.to(device) for key, value in batch_targets.items()}
                    target_schema = _structured_target_schema(targets)
                elif target_mode == "tensor":
                    if not isinstance(batch_targets, torch.Tensor):
                        raise TypeError(
                            "Tensor evaluation helpers require tensor batch targets, got "
                            f"{type(batch_targets)!r}."
                        )
                    targets = batch_targets.to(device)
                    target_schema = None
                else:
                    raise ValueError(
                        "evaluate_loader only supports target_mode='tensor' or target_mode='mapping', "
                        f"got {target_mode!r}."
                    )
                outputs = model(batch_x)
                validate_model_output(
                    model,
                    outputs,
                    batch_size=int(batch_x.shape[0]),
                    target_schema=target_schema,
                )
                if resolve_prediction_target is not None:
                    prediction, target = resolve_prediction_target(outputs, targets)
                else:
                    prediction = None
                    target = None
                if compute_loss is not None:
                    total_loss += float(compute_loss(outputs, targets, prediction, target).item())
                batches += 1
                samples += int(batch_x.shape[0])
                if prediction is not None and target is not None:
                    prediction_batches.append(prediction.detach().cpu())
                    target_batches.append(target.detach().cpu())

        return ForwardEvaluation(
            loss=(total_loss / max(1, batches)) if compute_loss is not None else None,
            batches=batches,
            samples=samples,
            predictions=torch.cat(prediction_batches, dim=0) if prediction_batches else None,
            targets=torch.cat(target_batches, dim=0) if target_batches else None,
        )
    finally:
        model.train(was_training)


def evaluate_tensor_target_loader(
    *,
    model: nn.Module,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loader: DataLoader | None,
) -> LossEvaluation:
    evaluation = evaluate_loader(
        model=model,
        device=device,
        loader=loader,
        target_mode="tensor",
        resolve_prediction_target=lambda outputs, targets: (outputs, targets),
        compute_loss=lambda _outputs, _targets, prediction, target: loss_fn(prediction, target),
    )
    return LossEvaluation(
        loss=float(evaluation.loss or 0.0),
        batches=evaluation.batches,
        samples=evaluation.samples,
    )


def evaluate_mapping_target_loader(
    *,
    model: nn.Module,
    device: torch.device,
    loss_fn: Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor],
    loader: DataLoader | None,
) -> LossEvaluation:
    """Evaluate mapping targets via ``loss_fn(outputs, targets)`` without narrowing keys.

    ``loss_fn`` receives the full structured outputs/targets mapping and must return
    one scalar tensor per batch. ``LossEvaluation.loss`` is the mean of those batch
    scalar values, not a per-field average across mapping entries.
    """
    evaluation = evaluate_loader(
        model=model,
        device=device,
        loader=loader,
        target_mode="mapping",
        resolve_prediction_target=None,
        compute_loss=lambda outputs, targets, _prediction, _target: loss_fn(outputs, targets),
    )
    return LossEvaluation(
        loss=float(evaluation.loss or 0.0),
        batches=evaluation.batches,
        samples=evaluation.samples,
    )


__all__ = [
    "ForwardEvaluation",
    "LossEvaluation",
    "evaluate_loader",
    "evaluate_mapping_target_loader",
    "evaluate_tensor_target_loader",
]
