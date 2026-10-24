import pandas as pd
import pytest

from star_agentic_rag.io_utils import resolve_input_columns, _normalize_question_type
from star_agentic_rag.sandbox import _check_imports, _extract_code_block


def test_resolve_input_columns_is_case_and_punctuation_insensitive():
    frame = pd.DataFrame(columns=["Question Text", "Gold Answer"])
    with pytest.raises(ValueError):
        resolve_input_columns(frame)
    frame = pd.DataFrame(columns=["QUESTION", "gold_answer"])
    q, a = resolve_input_columns(frame)
    assert q == "QUESTION"
    assert a == "gold_answer"


def test_question_type_normalization():
    assert _normalize_question_type("Single choice") == "Single choice"
    assert _normalize_question_type("Calculation") == "Calculation"
    assert _normalize_question_type("other") == "Unknown"


def test_import_allowlist_blocks_external_imports():
    assert _check_imports("import math\nprint(math.pi)")[0]
    ok, message = _check_imports("import os\nprint('hello')")
    assert not ok
    assert "os" in message


def test_code_block_extraction():
    assert _extract_code_block("```python\nprint(2 + 2)\n```") == "print(2 + 2)"
    assert _extract_code_block("nothing here") == ""


def test_sandbox_executes_simple_calculation():
    from star_agentic_rag.sandbox import _execute_python_sandbox

    output, success = _execute_python_sandbox("print(6 * 7)")
    assert success
    assert output == "42"


def test_llm_call_uses_ollama_compatible_token_parameters():
    from types import SimpleNamespace
    from star_agentic_rag.llm import _llm_call

    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=" ready "))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = _llm_call(client, "local-model", "hello", max_tokens=23, provider="ollama")

    assert result == "ready"
    assert captured["max_tokens"] == 23
    assert captured["extra_body"]["options"]["num_predict"] == 23
