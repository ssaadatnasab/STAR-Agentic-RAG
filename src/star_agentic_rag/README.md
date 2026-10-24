# STAR Agentic RAG

[← Repository overview](../../README.md) · [Inference guide](../../03_star_agentic_rag/README.md)

This package implements the complete Sense–Triage–Act–Reflect workflow.

| Module | Responsibility |
|---|---|
| [cli.py](cli.py) | CLI arguments, workbook processing, and checkpoint orchestration |
| [pipeline.py](pipeline.py) | LangGraph nodes, conditional routing, retrieval refinement, and answer verification |
| [retrieval.py](retrieval.py) | Corpus loading, BM25/dense retrieval, fusion, MMR, and reranking |
| [prompts.py](prompts.py) | Classification, triage, generation, and reflection prompts |
| [config.py](config.py) | Default paths, models, retrieval settings, and corpus routing |
| [state.py](state.py) | Shared graph state schema |
| [llm.py](llm.py) | Compatible API/Ollama clients and call handling |
| [sandbox.py](sandbox.py) | Generated Python extraction, basic checks, and subprocess execution |
| [io_utils.py](io_utils.py) | Workbook columns and saving |
| [response_parsing.py](response_parsing.py) | Structured response parsing |
| [__main__.py](__main__.py) | Module entry point |

## Run

After installing from the repository root:

```bash
star-agentic-rag --help
python -m star_agentic_rag --help
```

Follow the [inference guide](../../03_star_agentic_rag/README.md) for required workbooks, corpora, provider configuration, and output interpretation. Execution safeguards are described in [SECURITY.md](../../SECURITY.md).
