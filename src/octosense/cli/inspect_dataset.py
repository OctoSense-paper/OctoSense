"""CLI for dataset inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from octosense import datasets
from octosense.core.describe import DescribeNode


def inspect_dataset(
    dataset_id: str,
    *,
    modalities: Sequence[str],
    variant: str | None = None,
    split_scheme: str | None = None,
    task_binding: str | None = None,
    path: str | Path | None = None,
    split: str | None = None,
) -> dict[str, object]:
    view = datasets.load(
        dataset_id,
        modalities=modalities,
        variant=variant,
        split_scheme=split_scheme,
        task_binding=task_binding,
        path=path,
    )
    if split is not None:
        view = view.get_split(split)
    payload = {
        "dataset_id": dataset_id,
        "modalities": list(modalities),
        "variant": variant,
        "split_scheme": split_scheme,
        "task_binding": task_binding,
        "path": str(path) if path is not None else None,
        "split": split,
        "view_type": type(view).__name__,
        "sample_count": len(view),
    }
    describe = getattr(view, "describe_tree", None)
    if callable(describe):
        payload["describe_tree"] = describe().to_dict()
    return payload


def _render_text_payload(payload: dict[str, object]) -> str:
    lines = [
        f"dataset_id: {payload['dataset_id']}",
        f"modalities: {', '.join(payload['modalities'])}",
        f"variant: {payload['variant'] or '-'}",
        f"split_scheme: {payload['split_scheme'] or '-'}",
        f"task_binding: {payload['task_binding'] or '-'}",
        f"path: {payload['path'] or '-'}",
        f"split: {payload['split'] or '-'}",
        f"view_type: {payload['view_type']}",
        f"sample_count: {payload['sample_count']}",
    ]
    describe_tree = payload.get("describe_tree")
    if isinstance(describe_tree, dict):
        lines.extend(
            [
                "describe_tree:",
                DescribeNode.from_dict(describe_tree).render(),
            ]
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a canonical dataset bundle")
    parser.add_argument("--dataset-id", required=True, help="Canonical dataset identifier")
    parser.add_argument(
        "--modalities",
        nargs="+",
        required=True,
        help="Canonical modality list passed through datasets.load(...)",
    )
    parser.add_argument("--variant", default=None, help="Canonical dataset variant")
    parser.add_argument("--split-scheme", default=None, help="Canonical dataset split scheme")
    parser.add_argument("--task-binding", default=None, help="Canonical dataset task binding")
    parser.add_argument("--path", default=None, help="Optional dataset root override")
    parser.add_argument("--split", default=None, help="Optional split to materialize via DatasetView.get_split(...)")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = inspect_dataset(
        args.dataset_id,
        modalities=args.modalities,
        variant=args.variant,
        split_scheme=args.split_scheme,
        task_binding=args.task_binding,
        path=args.path,
        split=args.split,
    )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_text_payload(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
