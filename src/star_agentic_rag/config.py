"""Configuration defaults for STAR Agentic RAG.

Paths are repository-relative examples, not bundled corpora. Supply your own corpora.
"""
from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = "results/answers.xlsx"
DEFAULT_CORPUS = "data/corpora/markdown-aware-atomic-block-protection"
DEFAULT_MODEL_OPENAI = "gpt-5.5"
DEFAULT_MODEL_OLLAMA = "llama3.1:70b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BM25_TOP_K = 100
DEFAULT_DENSE_TOP_K = 100
DEFAULT_FUSED_TOP_K = 50
DEFAULT_MMR_TARGET_K = 20
DEFAULT_RAG_K = 5
MAX_RETRIEVAL_ATTEMPTS = 3
MAX_REFLECTION_CYCLES = 2
CODE_EXECUTION_TIMEOUT = 15
MAX_CODE_GEN_ATTEMPTS = 2
SAVE_EVERY_N_ROWS = 5

DEFAULT_FORMULA_CORPUS = "data/corpora/formula"
DEFAULT_IMAGE_CORPUS = "data/corpora/image"
DEFAULT_TABLE_CORPUS = "data/corpora/table"
DEFAULT_STAGE0_5_RAG_K = 5
DEFAULT_STAGE0_5_BM25_TOP_K = 100
DEFAULT_STAGE0_5_DENSE_TOP_K = 100
DEFAULT_STAGE0_5_FUSED_TOP_K = 50
DEFAULT_STAGE0_5_MMR_TARGET_K = 20
STAGE0_5_MAX_CONTEXT_CHARS = 4000
STAGE0_5_EXTRACTION_INPUT_CAP = 8000

CONTENT_TYPE_AWARE_CORPUS = "data/corpora/content-type-aware"
PARENT_CHILD_CORPUS = "data/corpora/parent-child"
ATOMIC_BLOCK_CORPUS = "data/corpora/markdown-aware-atomic-block-protection"
QUESTION_TYPE_TO_STRATEGY = {
    "Cloze": "parent_child",
    "Single choice": "atomic_block_protected",
    "Calculation": "atomic_block_protected",
    "Multi-choice": "content_type_aware",
    "Short Answer": "atomic_block_protected",
}
VALID_QUESTION_TYPES = list(QUESTION_TYPE_TO_STRATEGY)
ALLOWED_IMPORTS = {
    "math", "statistics", "decimal", "fractions", "datetime",
    "itertools", "collections", "cmath", "array",
}
