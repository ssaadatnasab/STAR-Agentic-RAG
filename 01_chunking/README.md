# Stage 1: Corpus Chunking

[← Repository overview](../README.md)

Build four alternative retrieval corpora from prepared Markdown documents. Each strategy preserves a different kind of document context.

| Script | Strategy |
|---|---|
| [chunk_hierarchical.py](chunk_hierarchical.py) | Splits along document headings and subdivides oversized sections |
| [chunk_content_type_aware.py](chunk_content_type_aware.py) | Handles text, formulas, and tables by content type |
| [chunk_parent_child.py](chunk_parent_child.py) | Builds small retrieval chunks linked to larger parent contexts |
| [chunk_atomic_block_protection.py](chunk_atomic_block_protection.py) | Protects atomic formula and table blocks during splitting |

## Setup and input

Install the root package, then the chunking utilities:

```bash
python -m pip install langchain-text-splitters tiktoken
```

Input is a directory of Markdown documents prepared upstream. For PDF extraction and enrichment, see [MRP-MSHRAG](https://github.com/ssaadatnasab/MRP-MSHRAG).

## Usage

Run from the repository root, replacing the paths with your corpus locations:

```bash
python 01_chunking/chunk_hierarchical.py --input-dir data/markdown --output-dir data/corpora/hierarchical
python 01_chunking/chunk_content_type_aware.py --input data/markdown --output data/corpora/content_type_aware
python 01_chunking/chunk_parent_child.py --input data/markdown --output data/corpora/parent_child
python 01_chunking/chunk_atomic_block_protection.py --input data/markdown --output data/corpora/atomic_block
```

Flag names differ by script; use `--help` for the accepted options. Some chunk-size settings are constants in the source.

## Output and next step

Each strategy writes its chunk corpus to the selected output directory. Parent-child retrieval uses `strategy3_children.jsonl` and `strategy3_parents.jsonl`. Use these corpora with [baseline inference](../02_baselines/README.md) or [STAR inference](../03_star_agentic_rag/README.md). Image, formula, and table corpora for specialized triage are supplied separately.
