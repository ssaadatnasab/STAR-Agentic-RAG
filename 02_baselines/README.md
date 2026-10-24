# Stage 2: LLM and Single-Pass RAG Baselines

[← Repository overview](../README.md)

Run no-retrieval inference or hybrid retrieval against the chunk corpora to compare with STAR Agentic RAG.

| Script | What it does |
|---|---|
| [llm_base_inference.py](llm_base_inference.py) | LLM inference without retrieval |
| [hybrid_rag_api.py](hybrid_rag_api.py) | Flat-corpus hybrid RAG through an API |
| [hybrid_rag_parent_child_api.py](hybrid_rag_parent_child_api.py) | Searches children and returns parent context through an API |
| [hybrid_rag_ollama.py](hybrid_rag_ollama.py) | Flat-corpus hybrid RAG through Ollama |
| [hybrid_rag_parent_child_ollama.py](hybrid_rag_parent_child_ollama.py) | Parent-child hybrid RAG through Ollama |

Hybrid retrieval combines BM25, dense FAISS retrieval, score fusion, MMR diversification, and cross-encoder reranking.

## Setup and inputs

Install the root package. Supply a question workbook and, for RAG, the corpus created in [Stage 1](../01_chunking/README.md). The scripts retain their original provider and machine-path defaults. Check each script's `--help`, set the model and endpoint, and supply explicit file paths. The API scripts expose `--key`; pass a key from a local environment variable. Not every variant reads credentials automatically from the environment or exposes a `--base_url` option.

## Usage

Run from the repository root:

```bash
python 02_baselines/llm_base_inference.py \
  --input data/questions.xlsx --output results/llm.xlsx \
  --model YOUR_MODEL_NAME --key "$OPENAI_API_KEY" \
  --base_url "$OPENAI_BASE_URL"

python 02_baselines/hybrid_rag_api.py \
  --input data/questions.xlsx --output results/rag.xlsx \
  --corpus data/corpora/atomic_block --model YOUR_MODEL_NAME \
  --key "$OPENAI_API_KEY" --base_url "$OPENAI_BASE_URL"

python 02_baselines/hybrid_rag_parent_child_ollama.py \
  --input data/questions.xlsx --output results/rag-parent-child.xlsx \
  --corpus data/corpora/parent_child --model YOUR_LOCAL_MODEL --key ollama
```

Set `OPENAI_BASE_URL` to your compatible endpoint before using the API examples. Start Ollama and provision your chosen model before the local example. Retrieval options include `--rag_k`, `--bm25_top_k`, `--dense_top_k`, `--fused_top_k`, `--mmr_lambda`, and `--mmr_target_k`.

## Outputs

Each run writes an Excel workbook with generated answers in `Result`; RAG variants also record retrieval context. Use the same questions and model settings for comparisons and pass the result workbook to [Stage 4: Evaluation](../04_evaluation/README.md).
