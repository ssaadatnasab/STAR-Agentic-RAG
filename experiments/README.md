# Research experiment scripts

These are the original standalone Python scripts supplied with the project, with embedded API key literals removed. They cover four chunking approaches, baseline inference, hybrid BM25/dense FAISS retrieval (including parent-child and Ollama variants), and Qwen judging. They are separate from the installable `src/scout_agentic_rag` package.

Many scripts have machine-specific `/home/user/Desktop/...` defaults. Provide your own input, output, corpus, model, and provider parameters after checking each script's `--help`. Install dependencies from the root `pyproject.toml` and any additional imports used by the chosen script. Do not commit credentials, benchmark data, or copyrighted source documents.

The scripts are research artifacts; their exact evaluations and numerical results have not been reproduced here. The key literals in the supplied originals should be treated as exposed and rotated with their provider.
