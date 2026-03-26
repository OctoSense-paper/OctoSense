"""Private OctoSense runtime internals.

This package intentionally avoids package-level re-exports so private helpers do
not look like a stable public API. Internal call sites should import concrete
helpers from the sibling modules that own them.
"""

__all__: list[str] = []
