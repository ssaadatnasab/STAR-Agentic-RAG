"""Command-line interface for processing question spreadsheets."""
import argparse
import os
import random
import time
from pathlib import Path
import pandas as pd

from .config import (
    ATOMIC_BLOCK_CORPUS, CONTENT_TYPE_AWARE_CORPUS, DEFAULT_CORPUS,
    DEFAULT_EMBEDDING_MODEL, DEFAULT_FORMULA_CORPUS, DEFAULT_IMAGE_CORPUS,
    DEFAULT_MODEL_OLLAMA, DEFAULT_MODEL_OPENAI, DEFAULT_OUTPUT,
    DEFAULT_RAG_K, DEFAULT_RERANKER_MODEL, DEFAULT_STAGE0_5_BM25_TOP_K,
    DEFAULT_STAGE0_5_DENSE_TOP_K, DEFAULT_STAGE0_5_FUSED_TOP_K,
    DEFAULT_STAGE0_5_MMR_TARGET_K, DEFAULT_STAGE0_5_RAG_K,
    DEFAULT_TABLE_CORPUS, DEFAULT_BM25_TOP_K, DEFAULT_DENSE_TOP_K,
    DEFAULT_FUSED_TOP_K, DEFAULT_MMR_TARGET_K, PARENT_CHILD_CORPUS,
    SAVE_EVERY_N_ROWS,
)
from .io_utils import _save_output, resolve_input_columns
from .llm import create_client
from .pipeline import make_graph
from .retrieval import _build_retriever_safe, build_hybrid_retriever
from .state import AgentState

def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG + Code — Pipeline 3.5 Scout-Agentic RAG (LangGraph)"
    )
    parser.add_argument("--input", required=True, help="Input .xlsx file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--provider", choices=["openai-compatible", "ollama"],
                        default=os.getenv("AGENTIC_RAG_PROVIDER", "openai-compatible"))
    parser.add_argument("--base-url", default=None,
                        help="Optional endpoint override; otherwise environment/default is used")
    parser.add_argument("--model", default=None, help="Model name (defaults depend on provider)")
    parser.add_argument("--rag_k",             type=int,   default=DEFAULT_RAG_K)
    parser.add_argument("--embedding_model",   default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker_model",    default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--bm25_top_k",        type=int,   default=DEFAULT_BM25_TOP_K)
    parser.add_argument("--dense_top_k",       type=int,   default=DEFAULT_DENSE_TOP_K)
    parser.add_argument("--fused_top_k",       type=int,   default=DEFAULT_FUSED_TOP_K)
    parser.add_argument("--mmr_target_k",      type=int,   default=DEFAULT_MMR_TARGET_K)
    parser.add_argument("--mmr_lambda",        type=float, default=0.7)
    parser.add_argument("--max_context_chars", type=int,   default=8000)
    parser.add_argument("--start_row", "--start-row", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=None, help="Checkpoint interval (defaults to 1 row for Ollama, 5 for OpenAI-compatible)")

    # Node 2 — Mixture-of-Experts chunking-strategy corpora
    parser.add_argument("--content_type_aware_corpus", default=CONTENT_TYPE_AWARE_CORPUS,
                         help="Expert corpus for multi-choice questions, as defined by the routing map.")
    parser.add_argument("--parent_child_corpus",       default=PARENT_CHILD_CORPUS,
                         help="Expert corpus for cloze questions, as defined by the routing map.")
    parser.add_argument("--atomic_block_corpus",       default=ATOMIC_BLOCK_CORPUS,
                         help="Expert corpus for single-choice, calculation, and short-answer questions.")

    # Stage 0.5 — specialized corpora
    parser.add_argument("--formula_corpus",         default=DEFAULT_FORMULA_CORPUS)
    parser.add_argument("--image_corpus",           default=DEFAULT_IMAGE_CORPUS)
    parser.add_argument("--table_corpus",           default=DEFAULT_TABLE_CORPUS)
    parser.add_argument("--stage0_5_rag_k",         type=int,   default=DEFAULT_STAGE0_5_RAG_K)
    parser.add_argument("--stage0_5_bm25_top_k",    type=int,   default=DEFAULT_STAGE0_5_BM25_TOP_K)
    parser.add_argument("--stage0_5_dense_top_k",   type=int,   default=DEFAULT_STAGE0_5_DENSE_TOP_K)
    parser.add_argument("--stage0_5_fused_top_k",   type=int,   default=DEFAULT_STAGE0_5_FUSED_TOP_K)
    parser.add_argument("--stage0_5_mmr_target_k",  type=int,   default=DEFAULT_STAGE0_5_MMR_TARGET_K)
    args = parser.parse_args()
    if args.save_every < 1:
        parser.error("--save-every must be at least 1")
    if args.model is None:
        args.model = DEFAULT_MODEL_OLLAMA if args.provider == "ollama" else DEFAULT_MODEL_OPENAI
    if args.save_every is None:
        args.save_every = 1 if args.provider == "ollama" else SAVE_EVERY_N_ROWS

    df = pd.read_excel(args.input)
    question_col, answer_col = resolve_input_columns(df)

    defaults = {
        "Result"                     : "",
        "RAG_Context"                : "",
        "query_type"                 : "",
        "question_type"              : "",
        "retrieval_strategy"         : "",
        # Stage 0.5 columns
        "needs_images"               : False,
        "needs_formulas"             : False,
        "needs_tables"               : False,
        "Stage0_5_RAG_Context"       : "",
        "key_info_extracted"         : "",
        "augmented_question"         : "",
        "generated_code"             : "",
        "code_execution_result"      : "",
        "code_execution_success"     : False,
        "code_generation_attempts"   : 0,
        "result_verification_passed" : False,
        "result_verification_notes"  : "",
        "query_complexity"           : "",
        "retrieval_attempts"         : 0,
        "reflection_cycles"          : 0,
        "relevance_verdict"          : "",
        "missing_info"               : "",
        "query_refined"              : False,
        "refined_query_text"         : "",
        "reflection_passed"          : False,
        "reflection_critique"        : "",
        "low_confidence"             : False,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)

    start_pos = max(1, args.start_row) - 1
    if start_pos >= len(df):
        print(f"Start row {args.start_row} exceeds total rows {len(df)}.", flush=True)
        _save_output(df, question_col, answer_col, args.output)
        return

    client = create_client(args.provider, args.base_url)
    retriever = build_hybrid_retriever(
        corpus_path=args.corpus,     k=args.rag_k,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        bm25_top_k=args.bm25_top_k, dense_top_k=args.dense_top_k,
        fused_top_k=args.fused_top_k, mmr_lambda=args.mmr_lambda,
        mmr_target_k=args.mmr_target_k,
    )

    print("🧪 Building Node 2 mixture-of-experts retrievers...", flush=True)
    content_type_aware_retriever = _build_retriever_safe(
        "content_type_aware", args.content_type_aware_corpus, args.rag_k, args.embedding_model,
        args.reranker_model, args.bm25_top_k, args.dense_top_k,
        args.fused_top_k, args.mmr_lambda, args.mmr_target_k,
    )
    parent_child_retriever = _build_retriever_safe(
        "parent_child", args.parent_child_corpus, args.rag_k, args.embedding_model,
        args.reranker_model, args.bm25_top_k, args.dense_top_k,
        args.fused_top_k, args.mmr_lambda, args.mmr_target_k,
    )
    atomic_block_retriever = _build_retriever_safe(
        "atomic_block_protected", args.atomic_block_corpus, args.rag_k, args.embedding_model,
        args.reranker_model, args.bm25_top_k, args.dense_top_k,
        args.fused_top_k, args.mmr_lambda, args.mmr_target_k,
    )
    main_retrievers = {
        "content_type_aware":     content_type_aware_retriever,
        "parent_child":           parent_child_retriever,
        "atomic_block_protected": atomic_block_retriever,
        "default":                retriever,
    }
    print(
        f"🧪 Node 2 retrievers ready — content_type_aware:{'OK' if content_type_aware_retriever else 'UNAVAILABLE'} "
        f"parent_child:{'OK' if parent_child_retriever else 'UNAVAILABLE'} "
        f"atomic_block_protected:{'OK' if atomic_block_retriever else 'UNAVAILABLE'} "
        f"default:{'OK' if retriever else 'UNAVAILABLE'}",
        flush=True,
    )

    # ── Stage 0.5 — build the three specialized-corpus retrievers ────────────
    # Each degrades independently and loudly (None + warning) rather than
    # crashing the whole run or silently pretending the corpus is empty.
    print("🧭 Building Stage 0.5 multimodal retrievers...", flush=True)
    image_retriever = _build_retriever_safe(
        "image", args.image_corpus, args.stage0_5_rag_k, args.embedding_model,
        args.reranker_model, args.stage0_5_bm25_top_k, args.stage0_5_dense_top_k,
        args.stage0_5_fused_top_k, args.mmr_lambda, args.stage0_5_mmr_target_k,
    )
    formula_retriever = _build_retriever_safe(
        "formula", args.formula_corpus, args.stage0_5_rag_k, args.embedding_model,
        args.reranker_model, args.stage0_5_bm25_top_k, args.stage0_5_dense_top_k,
        args.stage0_5_fused_top_k, args.mmr_lambda, args.stage0_5_mmr_target_k,
    )
    table_retriever = _build_retriever_safe(
        "table", args.table_corpus, args.stage0_5_rag_k, args.embedding_model,
        args.reranker_model, args.stage0_5_bm25_top_k, args.stage0_5_dense_top_k,
        args.stage0_5_fused_top_k, args.mmr_lambda, args.stage0_5_mmr_target_k,
    )
    multimodal_retrievers = {
        "image": image_retriever, "formula": formula_retriever, "table": table_retriever,
    }
    print(
        f"🧭 Stage 0.5 retrievers ready — image:{'OK' if image_retriever else 'UNAVAILABLE'} "
        f"formula:{'OK' if formula_retriever else 'UNAVAILABLE'} "
        f"table:{'OK' if table_retriever else 'UNAVAILABLE'}",
        flush=True,
    )

    agent = make_graph(client, main_retrievers, args.model, args.max_context_chars,
                        multimodal_retrievers=multimodal_retrievers, provider=args.provider)
    print(
        "✅ Pipeline 3.5 (Scout-Agentic RAG) graph compiled:\n"
        "   [factual]       classify_type → multimodal_triage (Stage 0.5) → retrieve → relevance "
        "→ [rewrite→] generate → reflect → [refine→] END\n"
        "   [computational] classify_type → retrieve (Stage 0.5 bypassed) → relevance → [rewrite→] "
        "generate_code → execute → verify → [retry_code | format | fallback_text] → END",
        flush=True,
    )

    processed = 0
    # MEMORY OPT: Cleanup before processing begins
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    for pos, q in df.loc[start_pos:, question_col].items():
        existing = df.at[pos, "Result"]
        if isinstance(existing, str) and existing.strip():
            print(f"⏭️  Skipping row {pos+1} (already answered)", flush=True)
            continue

        q_text    = "" if pd.isna(q) else str(q)
        human_row = pos + 1
        print(f"\n🔄 Row {human_row} | Q: {q_text[:70]}{'...' if len(q_text) > 70 else ''}",
              flush=True)

        initial_state: AgentState = {
            "question"                   : q_text,
            "query_type"                 : "",
            "query_complexity"           : "",
            "needs_images"               : False,
            "needs_formulas"             : False,
            "needs_tables"               : False,
            "image_rag_context"          : "",
            "formula_rag_context"        : "",
            "table_rag_context"          : "",
            "stage0_5_rag_context"       : "",
            "key_info_extracted"         : "",
            "augmented_question"         : q_text,
            "retrieval_attempts"         : 0,
            "current_query"              : q_text,
            "retrieved_docs"             : [],
            "rag_context"                : "",
            "relevance_verdict"          : "",
            "missing_info"               : "",
            "query_refined"              : False,
            "refined_query_text"         : "",
            "answer"                     : "",
            "reflection_passed"          : False,
            "reflection_critique"        : "",
            "reflection_cycles"          : 0,
            "low_confidence"             : False,
            "generated_code"             : "",
            "code_execution_result"      : "",
            "code_execution_success"     : False,
            "code_generation_attempts"   : 0,
            "result_verification_passed" : False,
            "result_verification_notes"  : "",
        }

        final_state: dict = {}
        for attempt in range(5):
            try:
                final_state = agent.invoke(initial_state)
                break
            except Exception as e:
                wait = min(100, (2 ** attempt) + random.random())
                print(f"  ⚠️  Agent failed: {e}, retry {attempt+1}/5 in {wait:.1f}s",
                      flush=True)
                time.sleep(wait)

        stage0_5_ctx = final_state.get("stage0_5_rag_context", "")
        if not str(stage0_5_ctx).strip():
            stage0_5_ctx = "No Stage 0.5 multimodal context used."

        df.at[pos, "Result"]                      = final_state.get("answer", "")
        df.at[pos, "RAG_Context"]                 = final_state.get("rag_context", "")
        df.at[pos, "query_type"]                  = final_state.get("query_type", "")
        df.at[pos, "question_type"]               = final_state.get("question_type", "")
        df.at[pos, "retrieval_strategy"]          = final_state.get("retrieval_strategy", "")
        df.at[pos, "needs_images"]                = final_state.get("needs_images", False)
        df.at[pos, "needs_formulas"]              = final_state.get("needs_formulas", False)
        df.at[pos, "needs_tables"]                = final_state.get("needs_tables", False)
        df.at[pos, "Stage0_5_RAG_Context"]        = stage0_5_ctx
        df.at[pos, "key_info_extracted"]          = final_state.get("key_info_extracted", "")
        df.at[pos, "augmented_question"]          = final_state.get("augmented_question", "")
        df.at[pos, "generated_code"]              = final_state.get("generated_code", "")
        df.at[pos, "code_execution_result"]       = final_state.get("code_execution_result", "")
        df.at[pos, "code_execution_success"]      = final_state.get("code_execution_success", False)
        df.at[pos, "code_generation_attempts"]    = final_state.get("code_generation_attempts", 0)
        df.at[pos, "result_verification_passed"]  = final_state.get("result_verification_passed", False)
        df.at[pos, "result_verification_notes"]   = final_state.get("result_verification_notes", "")
        df.at[pos, "query_complexity"]            = final_state.get("query_complexity", "")
        df.at[pos, "retrieval_attempts"]          = final_state.get("retrieval_attempts", 0)
        df.at[pos, "relevance_verdict"]           = final_state.get("relevance_verdict", "")
        df.at[pos, "missing_info"]                = final_state.get("missing_info", "")
        df.at[pos, "query_refined"]               = final_state.get("query_refined", False)
        df.at[pos, "refined_query_text"]          = final_state.get("refined_query_text", "")
        df.at[pos, "reflection_passed"]           = final_state.get("reflection_passed", False)
        df.at[pos, "reflection_critique"]         = final_state.get("reflection_critique", "")
        df.at[pos, "low_confidence"]              = final_state.get("low_confidence", False)
        df.at[pos, "reflection_cycles"]           = final_state.get("reflection_cycles", 0)

        qt = df.at[pos, "query_type"]
        print(
            f"  ✅ Done | type={qt} | complexity={df.at[pos,'query_complexity']} "
            f"| stage0.5(img/formula/table)={df.at[pos,'needs_images']}/"
            f"{df.at[pos,'needs_formulas']}/{df.at[pos,'needs_tables']} "
            f"| attempts={df.at[pos,'retrieval_attempts']} "
            f"| code_attempts={df.at[pos,'code_generation_attempts']} "
            f"| code_ok={df.at[pos,'code_execution_success']} "
            f"| verified={df.at[pos,'result_verification_passed']} "
            f"| reflection_passed={df.at[pos,'reflection_passed']} "
            f"| low_confidence={df.at[pos,'low_confidence']}",
            flush=True,
        )

        # Running summary
        comp_mask = df["query_type"] == "computational"
        fact_mask = df["query_type"] == "factual"
        print(f"  📈 Running stats ({processed+1} rows processed):", flush=True)
        print(
            f"     Computational: {comp_mask.sum()} | "
            f"Factual: {fact_mask.sum()}",
            flush=True,
        )
        if fact_mask.sum() > 0:
            print(
                f"     Stage 0.5 usage — images: {(df.loc[fact_mask,'needs_images']==True).mean():.0%} | "
                f"formulas: {(df.loc[fact_mask,'needs_formulas']==True).mean():.0%} | "
                f"tables: {(df.loc[fact_mask,'needs_tables']==True).mean():.0%}",
                flush=True,
            )
        if comp_mask.sum() > 0:
            print(
                f"     Code success rate    : {df.loc[comp_mask,'code_execution_success'].mean():.0%}",
                flush=True,
            )
            print(
                f"     Verification pass rate: {df.loc[comp_mask,'result_verification_passed'].mean():.0%}",
                flush=True,
            )
            print(
                f"     Avg code gen attempts : {df.loc[comp_mask,'code_generation_attempts'].mean():.2f}",
                flush=True,
            )
        print(
            f"     Rewrite rate   : {(df['query_refined']==True).mean():.0%} | "
            f"Low confidence: {(df['low_confidence']==True).mean():.0%}",
            flush=True,
        )

        processed += 1
        if processed % args.save_every == 0 or pos == df.index[-1]:
            _save_output(df, question_col, answer_col, args.output)
            print(f"  💾 Checkpoint saved at row {human_row} (every {args.save_every} rows).", flush=True)

        # MEMORY OPT: Cleanup after each row to prevent memory accumulation
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    _save_output(df, question_col, answer_col, args.output)
    print(f"\n🏁 Processing complete. {processed} rows processed. Saved to {args.output}",
          flush=True)

if __name__ == "__main__":
    main()
