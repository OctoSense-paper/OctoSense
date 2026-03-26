# 00 Installation And Environment Bootstrap

This file is only for installation guidance. It does not include any notebook
feature examples.

Run the following commands from the repository root:

```bash
uv sync --extra notebooks
uv run python -c "import octosense as octo; print(octo.__file__)"
```
