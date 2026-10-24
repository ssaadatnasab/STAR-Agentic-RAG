# STAR-Agentic-RAG

A modular implementation of the STAR (Sense–Triage–Act–Reflect) Agentic RAG pipeline for building-energy-science question answering. The workflow combines question classification, question-type-aware retrieval, optional image/formula/table retrieval, query refinement, reflective generation, and code execution for computational questions.

> **Research code notice:** This repository is a refactor of research code, not a claim of production readiness. Validate results for your corpus, models, and benchmark before using them in research or engineering decisions.

The installable Python package and CLI retain the internal name `scout_agentic_rag` from the research implementation. The public repository name follows the paper's STAR terminology.

## Pipeline overview

1. Classify each question as factual or computational, estimate complexity, and label its question format.
2. For factual questions, triage image, formula, and table corpora and distill useful information.
3. Select a main corpus based on question type and run hybrid retrieval (dense + BM25, score fusion, MMR, and cross-encoder reranking).
4. Judge retrieval sufficiency and iteratively rewrite queries to target missing information.
5. Generate and reflect on factual answers; computational questions generate Python code, execute it with basic checks, verify the result, and format the answer.
6. Save answers and diagnostic columns to an Excel workbook with periodic checkpoints.

## Repository layout

```text
src/scout_agentic_rag/
  config.py       Portable defaults and routing constants
  prompts.py      LLM prompt templates
  state.py        LangGraph state schema
  response_parsing.py Structured response parsing
  io_utils.py     Excel input/output helpers
  llm.py          OpenAI-compatible / Ollama client and chat retries
  sandbox.py      Generated-code execution helper
  retrieval.py    Corpus loading and hybrid retrieval
  pipeline.py     LangGraph workflow and conditional routing
  cli.py          Command-line entry point
  __main__.py     Enables python -m scout_agentic_rag
```

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/ssaadatnasab/STAR-Agentic-RAG.git
cd STAR-Agentic-RAG
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If you prefer a requirements file, use `pip install -r requirements.txt`.

## Configure a provider

### OpenAI or another OpenAI-compatible API

Optionally copy `.env.example` to `.env` and edit it locally (the `.env` file is ignored by Git). You can also set variables in your shell. Never commit real credentials.

```bash
export OPENAI_API_KEY="your-key"
# Optional for a compatible third-party endpoint:
# export OPENAI_BASE_URL="https://your-endpoint.example/v1"
```

Run with your input workbook and corpus paths:

```bash
scout-agentic-rag   --provider openai-compatible   --input data/questions.xlsx   --output results/answers.xlsx   --corpus data/corpora/markdown-aware-atomic-block-protection   --model gpt-5.5
```

### Local Ollama

Start Ollama and pull a model available on your machine. The default endpoint is `http://127.0.0.1:11434/v1`.

```bash
ollama pull llama3.1:70b
scout-agentic-rag   --provider ollama   --input data/questions.xlsx   --output results/answers-ollama.xlsx   --corpus data/corpora/markdown-aware-atomic-block-protection   --model llama3.1:70b
```

You can also run the CLI with `python -m scout_agentic_rag ...`.

## Input workbook

The first sheet of the `.xlsx` input is loaded. A question column is required; accepted normalized names include `question`, `questions`, `prompt`, and `query`. A reference-answer column is also currently required; accepted names include `answer`, `answers`, `goldanswer`, `referenceanswer`, and `groundtruth`. The reference answer is carried into the output for evaluation; it is not used as a target during generation.

Rows with a non-empty `Result` value are skipped so an interrupted run can be resumed. Use `--start_row N` to start from a specific 1-indexed spreadsheet row, including the header row in the numbering convention used by the script.

## Corpus organization

Corpus files are not included. Supply corpus folders containing `.jsonl` chunk files and/or `.md` files. The parent-child corpus expects `strategy3_children.jsonl` and `strategy3_parents.jsonl`.

The main corpus plus three chunking-strategy expert corpora are configurable with:

- `--corpus`
- `--content_type_aware_corpus`
- `--parent_child_corpus`
- `--atomic_block_corpus`

The specialized Stage 0.5 corpora can be configured with `--image_corpus`, `--formula_corpus`, and `--table_corpus`. They are optional; missing corpora are reported as unavailable.

Common retrieval parameters include `--rag_k`, `--embedding_model`, `--reranker_model`, `--bm25_top_k`, `--dense_top_k`, `--fused_top_k`, `--mmr_target_k`, and `--mmr_lambda`. See `scout-agentic-rag --help` for the full list.

## Output

The output workbook preserves the input question/reference answer and adds `Result`, retrieval context, routing decisions, multimodal triage information, generated code and execution logs, verification outcomes, and reflection diagnostics. The output directory is created automatically. Checkpoints default to every five processed rows for OpenAI-compatible endpoints and every row for Ollama; change this with `--save-every`.

## Important limitations

- Model-generated answers and calculations can be wrong. The reflection and verification steps are automated checks, not guarantees of correctness.
- The code-execution helper is **not a hardened sandbox**. Do not execute untrusted generated code outside an independently isolated environment. See [SECURITY.md](SECURITY.md).
- Retrieval quality depends on how your source documents are parsed, chunked, indexed, and licensed.
- Hardware and model requirements vary. Embedding and reranker initialization can require substantial memory and may download model weights on first use.
- Avoid sharing proprietary or copyrighted corpus files unless you have permission to redistribute them.

## Research scripts and paper context

The `experiments/` directory preserves the supplied standalone scripts for chunking, baseline inference, single-pass hybrid RAG, parent-child variants, local Ollama variants, and Qwen judging. Embedded API keys have been removed. These scripts retain environment-specific path defaults and are provided as research references; adjust paths and model/provider settings before running them. See [experiments/README.md](experiments/README.md).

The associated manuscript is *From Naive RAG to Agentic RAG: A Multimodal STAR Agentic RAG Framework for Building Science Question Answering* (Saadatnasab et al.). It describes a knowledge base of 181 textbooks and a comparison of no-retrieval, single-pass RAG, and STAR across multiple models. The manuscript, benchmark questions, textbook corpus, and model outputs are not distributed here. The code has not been independently reproduced against the paper's reported results.

## Development

```bash
pytest
ruff check src tests
```

## License

MIT. See [LICENSE](LICENSE). This license applies to this repository's code; it does not grant rights to external corpora, standards, textbooks, model weights, or datasets.
