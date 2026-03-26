"""Run manifest schema owned by ``octosense.specs``."""

from dataclasses import dataclass

RUN_MANIFEST_KIND = "RunManifest"


@dataclass(slots=True)
class RunManifest:
    """Execution facts emitted after a benchmark run completes."""

    kind: str = RUN_MANIFEST_KIND
    run_id: str = ""
    spec_digest: str = ""
    dataset_digest: str = ""
    git_sha: str = ""
    seed: int | None = None
    device: str | None = None
    status: str = "completed"
    started_at: str | None = None
    finished_at: str | None = None
    artifact_root: str | None = None
    metrics_path: str | None = None
    environment_path: str | None = None
    timing_path: str | None = None
    protocol_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "spec_digest": self.spec_digest,
            "dataset_digest": self.dataset_digest,
            "git_sha": self.git_sha,
            "seed": self.seed,
            "device": self.device,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "artifact_root": self.artifact_root,
            "metrics_path": self.metrics_path,
            "environment_path": self.environment_path,
            "timing_path": self.timing_path,
            "protocol_path": self.protocol_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "RunManifest":
        if payload is None:
            return cls()
        return cls(
            kind=str(payload.get("kind", RUN_MANIFEST_KIND) or RUN_MANIFEST_KIND),
            run_id=str(payload.get("run_id", "") or ""),
            spec_digest=str(payload.get("spec_digest", "") or ""),
            dataset_digest=str(payload.get("dataset_digest", "") or ""),
            git_sha=str(payload.get("git_sha", "") or ""),
            seed=_optional_int(payload.get("seed")),
            device=_optional_str(payload.get("device")),
            status=str(payload.get("status", "completed") or "completed"),
            started_at=_optional_str(payload.get("started_at")),
            finished_at=_optional_str(payload.get("finished_at")),
            artifact_root=_optional_str(payload.get("artifact_root")),
            metrics_path=_optional_str(payload.get("metrics_path")),
            environment_path=_optional_str(payload.get("environment_path")),
            timing_path=_optional_str(payload.get("timing_path")),
            protocol_path=_optional_str(payload.get("protocol_path")),
        )


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


__all__ = ["RUN_MANIFEST_KIND", "RunManifest"]
