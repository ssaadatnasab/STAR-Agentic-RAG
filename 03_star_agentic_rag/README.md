# STAR Agentic RAG: Inference

[← Repository overview](../README.md)

Run the complete Sense–Triage–Act–Reflect workflow through the installed `star-agentic-rag` command or `python -m star_agentic_rag`. The implementation lives in [src/star_agentic_rag](../src/star_agentic_rag/README.md).

All commands below run from the repository root.

## Setup

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
star-agentic-rag \
  --provider openai-compatible \
  --input data/questions.xlsx \
  --output results/answers.xlsx \
  --corpus data/corpora/markdown-aware-atomic-block-protection \
  --model gpt-5.5
```

### Local Ollama

Start Ollama and pull a model available on your machine. The default endpoint is `http://127.0.0.1:11434/v1`.

```bash
ollama pull llama3.1:70b
star-agentic-rag \
  --provider ollama \
  --input data/questions.xlsx \
  --output results/answers-ollama.xlsx \
  --corpus data/corpora/markdown-aware-atomic-block-protection \
  --model llama3.1:70b
```

You can also run the CLI with `python -m star_agentic_rag ...`.

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

Common retrieval parameters include `--rag_k`, `--embedding_model`, `--reranker_model`, `--bm25_top_k`, `--dense_top_k`, `--fused_top_k`, `--mmr_target_k`, and `--mmr_lambda`. See `star-agentic-rag --help` for the full list.

## Output

The output workbook preserves the input question/reference answer and adds `Result`, retrieval context, routing decisions, multimodal triage information, generated code and execution logs, verification outcomes, and reflection diagnostics. The output directory is created automatically. Checkpoints default to every five processed rows for OpenAI-compatible endpoints and every row for Ollama; change this with `--save-every`.

## Important limitations

- Model-generated answers and calculations can be wrong. The reflection and verification steps are automated checks, not guarantees of correctness.
- The code-execution helper is **not a hardened sandbox**. Do not execute untrusted generated code outside an independently isolated environment. See [SECURITY.md](../SECURITY.md).
- Retrieval quality depends on how your source documents are parsed, chunked, indexed, and licensed.
- Hardware and model requirements vary. Embedding and reranker initialization can require substantial memory and may download model weights on first use.
- Avoid sharing proprietary or copyrighted corpus files unless you have permission to redistribute them.



