"""Dataset storage package.

Canonical dataset loading stays centered on ``octosense.datasets.load(...)``.
Storage implementation modules remain package-internal instead of exposing a
second importable control plane from ``octosense.datasets.storage``.
"""

__all__: list[str] = []
