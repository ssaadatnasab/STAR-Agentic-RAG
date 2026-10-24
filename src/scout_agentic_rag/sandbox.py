"""Helpers for executing model-generated calculation code.

IMPORTANT: the allowlist and subprocess timeout are defense-in-depth only; they are
not a hardened security boundary. Run untrusted generated code in a container/VM.
"""
import os
import re
import subprocess
import sys
import tempfile
from .config import ALLOWED_IMPORTS, CODE_EXECUTION_TIMEOUT

def _check_imports(code: str) -> tuple[bool, str]:
    """
    FIX Issue 2: Scan generated code for disallowed imports before execution.
    Returns (is_safe, error_message).
    """
    found = re.findall(r'^\s*import\s+(\w+)', code, re.MULTILINE)
    found += re.findall(r'^\s*from\s+(\w+)', code, re.MULTILINE)
    bad = set(found) - ALLOWED_IMPORTS
    if bad:
        return False, f"Blocked import(s): {bad}"
    return True, ""

def _extract_code_block(text: str) -> str:
    """
    Extract Python code from a ```python ... ``` fence.
    FIX Issue 6: last-resort path validates that text looks like Python
    before returning it; returns "" if it does not.
    """
    # Try ```python ... ``` first
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Try generic ``` ... ```
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # FIX Issue 6: only fall back to raw text if it contains Python syntax ────
    if re.search(r'\b(print|def |result\s*=|math\.)\b', text):
        return text.strip()
    return ""   # caller treats empty string as a generation failure

def _execute_python_sandbox(code: str,
                             timeout: int = CODE_EXECUTION_TIMEOUT) -> tuple[str, bool]:
    """
    Execute Python code in an isolated subprocess.
    FIX Issue 2: import allowlist checked before subprocess.run.
    Returns (output, success).
    """
    if not code.strip():
        return "ERROR: Empty code block received.", False

    # FIX Issue 2 ──────────────────────────────────────────────────────────────
    ok, msg = _check_imports(code)
    if not ok:
        return f"BLOCKED: {msg}", False
    BLOCKED_BUILTINS_RE = re.compile(r'\b(__import__|exec|eval|compile|open|__builtins__)\s*\(')
    if BLOCKED_BUILTINS_RE.search(code):
        return "BLOCKED: dangerous built-in call detected", False
    # ──────────────────────────────────────────────────────────────────────────

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write("import math\n")   # pre-import so LLM code can use math.*
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, "-I", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode != 0:
            return f"EXECUTION ERROR:\n{stderr or f'Exit code {result.returncode}'}", False
        if not stdout:
            return "Code ran without errors but produced no output.", True
        return stdout, True
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: Code exceeded {timeout}s execution limit.", False
    except Exception as e:
        return f"SANDBOX ERROR: {e}", False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
