# Tests

[← Repository overview](../README.md)

Install development dependencies from the repository root and run:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

[test_helpers.py](test_helpers.py) checks workbook-column matching, question-type normalization, code extraction, import checks, simple Python execution, and Ollama request parameters. The tests do not call a live model or reproduce the manuscript's benchmark.
