"""Input/output and tabular-data helpers."""
import re
import time
from pathlib import Path
import pandas as pd

ILLEGAL_EXCEL_CHAR_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

def _normalize_col(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())

def resolve_input_columns(df: pd.DataFrame) -> tuple[str, str]:
    normalized_map = {_normalize_col(c): c for c in df.columns}
    question_col = next(
        (normalized_map[k] for k in ["question", "questions", "prompt", "query"]
         if k in normalized_map), None
    )
    answer_col = next(
        (normalized_map[k] for k in ["answer", "answers", "goldanswer",
                                      "referenceanswer", "groundtruth"]
         if k in normalized_map), None
    )
    if question_col is None:
        raise ValueError(f"Input file missing question column. Detected: {list(df.columns)}")
    if answer_col is None:
        raise ValueError(f"Input file missing answer column. Detected: {list(df.columns)}")
    return question_col, answer_col

def _normalize_question_type(raw: str) -> str:
    """
    Map the question-type classifier's raw output to one of the five canonical
    labels used for Node 2's mixture-of-experts corpus routing. Falls back to
    "Unknown" if nothing matches, which routes to the default corpus.
    """
    r = raw.strip().lower()
    if "cloze" in r:
        return "Cloze"
    if "multi" in r and "choice" in r:
        return "Multi-choice"
    if "single" in r and "choice" in r:
        return "Single choice"
    if "calculat" in r:
        return "Calculation"
    if "short" in r:
        return "Short Answer"
    return "Unknown"

def _sanitize_excel_value(value):
    if isinstance(value, str):
        return ILLEGAL_EXCEL_CHAR_RE.sub("", value)
    return value

def _save_output(df: pd.DataFrame, question_col: str, answer_col: str, output_path: str):
    """Save results to Excel. FIX Issue 9: handles PermissionError with backup file."""
    cols = [
        question_col, answer_col, "RAG_Context", "Result",
        "query_type", "question_type", "retrieval_strategy",
        # Stage 0.5 columns
        "needs_images", "needs_formulas", "needs_tables",
        "Stage0_5_RAG_Context", "key_info_extracted", "augmented_question",
        "generated_code", "code_execution_result",
        "code_execution_success", "code_generation_attempts",
        "result_verification_passed", "result_verification_notes",
        "query_complexity", "retrieval_attempts", "reflection_cycles",
        "relevance_verdict", "missing_info",
        "query_refined", "refined_query_text",
        "reflection_passed", "reflection_critique", "low_confidence",
    ]
    out_df = df[[c for c in cols if c in df.columns]].rename(
        columns={question_col: "question", answer_col: "Answer"}
    ).copy()
    for col in out_df.columns:
        out_df[col] = out_df[col].map(_sanitize_excel_value)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_df.to_excel(output, index=False)
    except PermissionError:
        backup = str(output.with_name(f"{output.stem}_backup_{int(time.time())}{output.suffix}"))
        out_df.to_excel(backup, index=False)
        print(f"⚠️  Output file locked — saved backup to {backup}", flush=True)
