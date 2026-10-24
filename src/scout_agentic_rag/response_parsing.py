"""Parsers for structured model responses."""
import re

def _parse_reflection(critique: str) -> dict:
    """
    Parse structured reflection output.
    FIX Issue 3: returns {} when critique is empty; callers treat {} as failure.
    """
    if not critique.strip():
        return {}   # FIX Issue 3: empty → caller sets reflection_passed = False
    result = {}
    for line in critique.lower().splitlines():
        for key in ["addresses_question", "grounded_in_context", "units_and_magnitude_ok"]:
            if key in line:
                value_part = line.split(":", 1)[-1].strip() if ":" in line else line
                if "partial" in value_part:
                    result[key] = "partially"
                elif re.search(r'\bno\b', value_part):
                    result[key] = "no"
                else:
                    result[key] = "yes"
    return result
