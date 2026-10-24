"""Typed state passed between LangGraph nodes.
"""
from typing import TypedDict

class AgentState(TypedDict):
    # Input
    question: str
    # Query classification
    query_type: str                       # "computational" | "factual"
    query_complexity: str                 # "simple" | "complex"
    question_type: str                    # "Cloze" | "Single choice" | "Multi-choice" | "Calculation" | "Short Answer"
    retrieval_strategy: str               # which Node 2 chunking-strategy corpus was used
    # Stage 0.5 — multimodal triage (factual path only)
    needs_images: bool
    needs_formulas: bool
    needs_tables: bool
    image_rag_context: str
    formula_rag_context: str
    table_rag_context: str
    stage0_5_rag_context: str             # combined raw context (for logging/Excel)
    key_info_extracted: str               # distilled info actually used downstream
    augmented_question: str               # question + key_info_extracted
    # Retrieval tracking
    retrieval_attempts: int
    current_query: str
    retrieved_docs: list
    rag_context: str
    relevance_verdict: str
    missing_info: str
    query_refined: bool
    refined_query_text: str
    # Factual path — generation + reflection
    answer: str
    reflection_passed: bool
    reflection_critique: str
    reflection_cycles: int
    low_confidence: bool
    # Computational path — code execution
    generated_code: str
    code_execution_result: str
    code_execution_success: bool
    code_generation_attempts: int         # FIX Issue 7: tracks retry count
    result_verification_passed: bool
    result_verification_notes: str
