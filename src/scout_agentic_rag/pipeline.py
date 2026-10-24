"""LangGraph orchestration for the Scout-Agentic RAG workflow."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import re
from typing import Literal, Optional

from .config import (
    CODE_EXECUTION_TIMEOUT, MAX_CODE_GEN_ATTEMPTS, MAX_REFLECTION_CYCLES,
    MAX_RETRIEVAL_ATTEMPTS, QUESTION_TYPE_TO_STRATEGY,
    STAGE0_5_EXTRACTION_INPUT_CAP, STAGE0_5_MAX_CONTEXT_CHARS,
)
from .io_utils import _normalize_question_type
from .llm import _llm_call, _web_search_snippets
from .prompts import (
    CLASSIFICATION_PROMPT, CODE_ANSWER_FORMAT_PROMPT, CODE_GENERATION_PROMPT,
    KEY_INFO_EXTRACTION_PROMPT, QUERY_REWRITE_PROMPT, QUESTION_TYPE_PROMPT,
    REFINE_PROMPT, REFLECTION_PROMPT, RELEVANCE_PROMPT, RESULT_VERIFICATION_PROMPT,
    SYSTEM_PROMPT, TRIAGE_PLANNER_PROMPT,
)
from .sandbox import _execute_python_sandbox, _extract_code_block
from .state import AgentState
from .retrieval import docs_to_context
from .response_parsing import _parse_reflection

def make_graph(client, main_retrievers: dict, model: str, max_context_chars: int,
               multimodal_retrievers: Optional[dict] = None,
               provider: str = "openai-compatible"):
    """
    multimodal_retrievers: dict with keys "image", "formula", "table", each mapping
    to a retrieve(query)->list[Document] function, or None if that corpus is
    unavailable (e.g. folder missing / failed to build). Missing retrievers
    degrade explicitly (logged warning + marker in saved context) rather than
    silently pretending "no info needed."
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        raise ImportError("Install langgraph: pip install langgraph")

    llm_call = partial(_llm_call, provider=provider)
    multimodal_retrievers = multimodal_retrievers or {}
    image_retriever   = multimodal_retrievers.get("image")
    formula_retriever = multimodal_retrievers.get("formula")
    table_retriever   = multimodal_retrievers.get("table")

    def _select_main_retriever(question_type: str):
        """
        Mixture-of-experts routing: pick the chunking-strategy corpus that
        performed best for this question_type on the 149-question benchmark.
        Falls back to "default" with a loud warning if needed.
        """
        strategy = QUESTION_TYPE_TO_STRATEGY.get(question_type)
        if strategy is None:
            print(f"  ⚠️  Node 2 — unrecognized question_type '{question_type}', falling back to default corpus", flush=True)
            return "default", main_retrievers.get("default")
        retriever_fn = main_retrievers.get(strategy)
        if retriever_fn is None:
            print(f"  ⚠️  Node 2 — expert corpus '{strategy}' for question_type '{question_type}' is unavailable, falling back to default corpus", flush=True)
            return "default", main_retrievers.get("default")
        return strategy, retriever_fn

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 0 — Unified Query Classifier
    # ══════════════════════════════════════════════════════════════════════════
    def classify_query_type(state: AgentState) -> AgentState:
        question = state["question"]
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_type = executor.submit(
                llm_call, client, model, CLASSIFICATION_PROMPT.format(question=question), "", 30
            )
            future_qtype = executor.submit(
                llm_call, client, model, QUESTION_TYPE_PROMPT.format(question=question), "", 20
            )
            raw = future_type.result().lower().strip()
            qtype_raw = future_qtype.result().strip()

        q_type = "factual"
        q_complexity = "complex"
        for line in raw.splitlines():
            if line.startswith("type"):
                vp = line.split(":", 1)[-1].strip()
                if vp in ("computational", "factual"):
                    q_type = vp
            elif line.startswith("complexity"):
                vp = line.split(":", 1)[-1].strip()
                if vp in ("simple", "complex"):
                    q_complexity = vp
        q_question_type = _normalize_question_type(qtype_raw)
        print(f"  🧮 Query type: {q_type} | complexity: {q_complexity} | question_type: {q_question_type}", flush=True)

        state["query_type"]                  = q_type
        state["query_complexity"]            = q_complexity
        state["question_type"]               = q_question_type
        state["retrieval_strategy"]          = ""
        state["current_query"]               = state["question"]
        state["retrieval_attempts"]          = 0
        state["query_refined"]               = False
        state["refined_query_text"]          = ""
        state["reflection_cycles"]           = 0
        state["low_confidence"]              = False
        state["generated_code"]              = ""
        state["code_execution_result"]       = ""
        state["code_execution_success"]      = False
        state["code_generation_attempts"]    = 0   # FIX Issue 7
        state["result_verification_passed"]  = False
        state["result_verification_notes"]   = ""
        # Stage 0.5 defaults (populated only if multimodal_triage runs)
        state["needs_images"]                = False
        state["needs_formulas"]              = False
        state["needs_tables"]                = False
        state["image_rag_context"]           = ""
        state["formula_rag_context"]         = ""
        state["table_rag_context"]           = ""
        state["stage0_5_rag_context"]        = ""
        state["key_info_extracted"]          = ""
        state["augmented_question"]          = state["question"]
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 0.5 — Multimodal Knowledge Triage  (ALL QUERY TYPES)
    # ══════════════════════════════════════════════════════════════════════════
    def multimodal_triage(state: AgentState) -> AgentState:
        question = state["question"]

        def _line_value(text: str, key: str) -> str:
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if line.lower().startswith(f"{key.lower()}:"):
                    return line.split(":", 1)[-1].strip()
            return ""

        def _is_yes(value: str) -> bool:
            v = (value or "").strip().lower()
            return bool(v) and ("yes" in v)

        # ── Step 1: LLM planner decides which specialized corpora are needed ──
        plan_raw = llm_call(
            client, model, TRIAGE_PLANNER_PROMPT.format(question=question), max_tokens=200
        )

        images_needed_raw = _line_value(plan_raw, "images_needed")
        formulas_needed_raw = _line_value(plan_raw, "formulas_needed")
        tables_needed_raw = _line_value(plan_raw, "tables_needed")

        image_query = _line_value(plan_raw, "images_query") or question
        formula_query = _line_value(plan_raw, "formulas_query") or question
        table_query = _line_value(plan_raw, "tables_query") or question

        image_needs = _line_value(plan_raw, "images_needs")
        formula_needs = _line_value(plan_raw, "formulas_needs")
        table_needs = _line_value(plan_raw, "tables_needs")

        needs_images = _is_yes(images_needed_raw)
        needs_formulas = _is_yes(formulas_needed_raw)
        needs_tables = _is_yes(tables_needed_raw)

        state["needs_images"]   = needs_images
        state["needs_formulas"] = needs_formulas
        state["needs_tables"]   = needs_tables
        print(
            f"  🧭 Stage 0.5 triage — images:{needs_images} formulas:{needs_formulas} "
            f"tables:{needs_tables}",
            flush=True,
        )
        print(
            f"  🔎 Stage 0.5 needs — image:[{image_needs or 'none'}] "
            f"formula:[{formula_needs or 'none'}] table:[{table_needs or 'none'}]",
            flush=True,
        )

        # ── Step 2: retrieve only from the corpora flagged "yes" ───────────────
        combined_segments = []

        def _retrieve_one(label: str, needed: bool, retriever_fn, query_text: str):
            if not needed:
                return ""
            if retriever_fn is None:
                # Explicit, loud degradation — never silently pretend "no info needed"
                print(f"  ⚠️  Stage 0.5 — {label} corpus needed but retriever is unavailable!",
                      flush=True)
                return f"=== {label.upper()} CORPUS ===\n[WARNING: {label} retriever unavailable — corpus could not be loaded]"
            docs = retriever_fn(query_text)
            ctx  = docs_to_context(docs, max_chars=STAGE0_5_MAX_CONTEXT_CHARS)
            print(
                f"  📥 Stage 0.5 — retrieved {len(docs)} chunks from {label} corpus "
                f"using query: {query_text[:90]}",
                flush=True,
            )
            return f"=== {label.upper()} CORPUS ===\n{ctx}" if ctx else ""

        image_ctx   = _retrieve_one("image",   needs_images,   image_retriever, image_query)
        formula_ctx = _retrieve_one("formula", needs_formulas, formula_retriever, formula_query)
        table_ctx   = _retrieve_one("table",   needs_tables,   table_retriever, table_query)

        state["image_rag_context"]   = image_ctx
        state["formula_rag_context"] = formula_ctx
        state["table_rag_context"]   = table_ctx

        for seg in (image_ctx, formula_ctx, table_ctx):
            if seg:
                combined_segments.append(seg)
        combined_context = "\n\n".join(combined_segments).strip()
        state["stage0_5_rag_context"] = combined_context

        # ── Step 3: LLM extracts ONLY the key info needed — never pass raw
        #            retrieved content downstream, it is too long. ────────────
        if combined_context:
            extraction_prompt = KEY_INFO_EXTRACTION_PROMPT.format(
                question=question,
                multimodal_context=combined_context[:STAGE0_5_EXTRACTION_INPUT_CAP],
            )
            key_info = llm_call(client, model, extraction_prompt, max_tokens=400)
        else:
            key_info = "No additional information needed."

        if not key_info.strip():
            key_info = "No additional information needed."

        state["key_info_extracted"] = key_info
        if (
            not state.get("stage0_5_rag_context", "").strip()
            and "no additional information needed" not in key_info.lower()
        ):
            state["stage0_5_rag_context"] = (
                "=== STAGE0_5_EXTRACTED_KEY_INFO_ONLY ===\n"
                f"{key_info.strip()}"
            )
        print(f"  🔎 Stage 0.5 — extracted key info ({len(key_info)} chars)", flush=True)

        # ── Step 4: keep augmented question for answering context, but
        #            keep Node 2 retrieval query as original question. ─────────
        if "no additional information needed" in key_info.lower():
            augmented_question = question
        else:
            augmented_question = (
                f"{question}\n\n"
                f"Additional Known Information (from image/formula/table corpora):\n{key_info}"
            )
        state["augmented_question"] = augmented_question
        state["current_query"]      = question
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 2a — Retrieve
    # ══════════════════════════════════════════════════════════════════════════
    def retrieve_docs(state: AgentState) -> AgentState:
        attempt = state["retrieval_attempts"] + 1
        strategy, retriever_fn = _select_main_retriever(state.get("question_type", "Unknown"))
        state["retrieval_strategy"] = strategy

        missing = (state.get("missing_info") or "").strip()
        focused_query = state["current_query"]
        if missing and missing.lower() not in {"none", "unknown"}:
            focused_query = f"{state['current_query']}\nFocus missing info: {missing}"

        selected_label = strategy
        selected_retriever = retriever_fn
        if attempt == 2 and table_retriever is not None:
            selected_label = f"{strategy}+table_retry"
            selected_retriever = table_retriever
        elif attempt >= 3 and formula_retriever is not None:
            selected_label = f"{strategy}+formula_retry+web"
            selected_retriever = formula_retriever

        state["retrieval_strategy"] = selected_label
        print(f"  📥 Retrieval attempt {attempt} | strategy: {selected_label} | query: {focused_query[:60]}...",
              flush=True)
        if selected_retriever is None:
            print("  ⚠️  Node 2 — no retriever available for this question at all — returning empty context", flush=True)
            docs, context = [], ""
        else:
            docs    = selected_retriever(focused_query)
            context = docs_to_context(docs, max_chars=max_context_chars)

        if attempt >= 3:
            web_hits = _web_search_snippets(focused_query, max_results=5)
            state["web_search_context"] = web_hits
            state["web_search_used"] = bool(web_hits)
            if web_hits:
                web_block = f"[Web Fallback Context]\n{web_hits}"
                context = (context + "\n\n" + web_block).strip()[:max_context_chars]
                print("  🌐 Web snippets appended to context", flush=True)

        state["retrieved_docs"]     = docs
        state["rag_context"]        = context
        state["retrieval_attempts"] = attempt
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 2b — Relevance Check
    # ══════════════════════════════════════════════════════════════════════════
    def check_relevance(state: AgentState) -> AgentState:
        prompt = RELEVANCE_PROMPT.format(
            question=state["question"],
            context=state["rag_context"][:3000],
        )
        raw     = llm_call(client, model, prompt, max_tokens=120).lower()
        verdict = "partial"
        missing = "unknown"
        for line in raw.splitlines():
            if line.startswith("verdict"):
                vp = line.split(":", 1)[-1].strip() if ":" in line else line
                if "partial" in vp:
                    verdict = "partial"
                elif re.search(r'\bno\b', vp):
                    verdict = "no"
                elif "yes" in vp:
                    verdict = "yes"
            elif line.startswith("missing"):
                missing = line.split(":", 1)[-1].strip()
        print(f"  📊 Relevance: {verdict} | missing: {missing[:80]}", flush=True)
        state["relevance_verdict"] = verdict
        state["missing_info"]      = missing
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 2c — Query Rewriter
    # ══════════════════════════════════════════════════════════════════════════
    def rewrite_query(state: AgentState) -> AgentState:
        context_summary = state["rag_context"][:800] if state["rag_context"] else "Nothing retrieved yet."
        missing_info    = state.get("missing_info", "not specified")
        prompt = QUERY_REWRITE_PROMPT.format(
            question=state["question"],
            attempt=state["retrieval_attempts"],
            context_summary=context_summary,
            missing_info=missing_info,
        )
        new_query = llm_call(client, model, prompt, max_tokens=320)
        if new_query:
            print(f"  ✏️  Refined query: {new_query[:60]}...", flush=True)
            state["current_query"]      = new_query
            state["query_refined"]      = True
            state["refined_query_text"] = new_query
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 3a — Generate Answer  (factual path)
    # ══════════════════════════════════════════════════════════════════════════
    def generate_answer(state: AgentState) -> AgentState:
        key_info = state.get("key_info_extracted", "")
        has_key_info = bool(key_info) and "no additional information needed" not in key_info.lower()
        key_info_block = (
            f"Additional Known Information (from image/formula/table corpora):\n{key_info}\n\n"
            if has_key_info else ""
        )
        user_prompt = (
            f"Question:\n{state['question']}\n\n"
            f"{key_info_block}"
            f"Retrieved Context (Agentic RAG):\n"
            f"{state['rag_context'] if state['rag_context'] else 'No relevant context found.'}\n\n"
            "Instruction:\nAnswer the question using the retrieved context (and the additional "
            "known information above, if provided) when relevant."
        )
        answer = llm_call(client, model, user_prompt, system=SYSTEM_PROMPT, max_tokens=1500)
        print(f"  ✍️  Generated answer ({len(answer)} chars)", flush=True)
        state["answer"] = answer
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 3b — Reflect  (factual path)
    # ══════════════════════════════════════════════════════════════════════════
    def reflect(state: AgentState) -> AgentState:
        prompt   = REFLECTION_PROMPT.format(
            question=state["question"],
            answer=state["answer"],
            context=state["rag_context"][:2000],
        )
        critique = llm_call(client, model, prompt, max_tokens=80)
        lines    = _parse_reflection(critique)

        # FIX Issue 3: empty critique → fail immediately ──────────────────────
        if not lines:
            print("  🪞 Reflection — empty critique → forcing refinement", flush=True)
            state["reflection_passed"]   = False
            state["reflection_critique"] = critique
            state["reflection_cycles"]   = state.get("reflection_cycles", 0) + 1
            return state
        # ──────────────────────────────────────────────────────────────────────

        # FIX Issue 3: defaults changed from "yes" to "no" / "na" ─────────────
        addresses = lines.get("addresses_question",     "no")
        grounded  = lines.get("grounded_in_context",    "no")
        units_ok  = lines.get("units_and_magnitude_ok", "na")
        # ──────────────────────────────────────────────────────────────────────

        passed = (
            addresses == "yes"
            and grounded in ("yes", "partially")
            and units_ok in ("yes", "na")
        )
        print(
            f"  🪞 Reflection — addresses:{addresses} grounded:{grounded} "
            f"units:{units_ok} → {'✅' if passed else '❌'}",
            flush=True,
        )
        state["reflection_passed"]   = passed
        state["reflection_critique"] = critique
        state["reflection_cycles"]   = state.get("reflection_cycles", 0) + 1
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 3c — Refine Answer  (factual path)
    # ══════════════════════════════════════════════════════════════════════════
    def refine_answer(state: AgentState) -> AgentState:
        prompt  = REFINE_PROMPT.format(
            question=state["question"],
            answer=state["answer"],
            critique=state["reflection_critique"],
            context=state["rag_context"],
        )
        refined = llm_call(client, model, prompt, system=SYSTEM_PROMPT, max_tokens=1500)
        if refined:
            print(f"  🔧 Answer refined ({len(refined)} chars)", flush=True)
            state["answer"] = refined
        re_check = llm_call(
            client, model,
            f"Is this answer correct and complete?\nQuestion: {state['question']}\n"
            f"Answer: {state['answer']}\nReply ONLY: yes or no",
            max_tokens=5,
        ).lower().strip()
        state["low_confidence"] = (re_check != "yes")
        print(f"  📋 Post-refine confidence: {re_check} → low_confidence={state['low_confidence']}",
              flush=True)
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 4a — Generate Code  (computational path)
    # FIX Issue 8: inject previous execution error on retry
    # ══════════════════════════════════════════════════════════════════════════
    def generate_code(state: AgentState) -> AgentState:
        attempts = state.get("code_generation_attempts", 0) + 1
        state["code_generation_attempts"] = attempts

        # FIX Issue 8: build error hint for retry attempts ────────────────────
        error_hint = ""
        prev_exec = state.get("code_execution_result", "")
        verify_note = state.get("result_verification_notes", "")
        if attempts > 1:
            if not state.get("code_execution_success", False):
                error_hint = (
                    f"EXECUTION ERROR ON PREVIOUS ATTEMPT:\n"
                    f"{prev_exec[:400]}\n"
                    f"Fix the error."
                )
            elif verify_note:
                error_hint = (
                    f"PREVIOUS RESULT FAILED VERIFICATION:\n"
                    f"{verify_note}\n"
                    f"Revise your approach."
                )
            print(f"  🔁 Code regeneration attempt {attempts} (with error feedback)", flush=True)
        error_hint = f"\n{error_hint}\n" if error_hint else ""
        # ──────────────────────────────────────────────────────────────────────

        prompt = CODE_GENERATION_PROMPT.format(
            question=state["question"],
            context=state["rag_context"] if state["rag_context"] else "No context retrieved.",
            error_hint=error_hint,
        )
        raw_response = llm_call(client, model, prompt, max_tokens=1200)
        code = _extract_code_block(raw_response)
        print(f"  💻 Code generated ({len(code)} chars)", flush=True)
        if len(code) < 20:
            print("  ⚠️  Code block too short — may be extraction failure", flush=True)
        state["generated_code"] = code
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 4b — Execute Code  (computational path)
    # ══════════════════════════════════════════════════════════════════════════
    def execute_code(state: AgentState) -> AgentState:
        code = state.get("generated_code", "")
        if not code.strip():
            print("  ⚠️  No code to execute — falling back to text generation", flush=True)
            state["code_execution_result"]  = "No code was generated."
            state["code_execution_success"] = False
            return state
        print("  ⚙️  Executing code in sandbox...", flush=True)
        output, success = _execute_python_sandbox(code, timeout=CODE_EXECUTION_TIMEOUT)
        print(f"  {'✅' if success else '❌'} Execution: {output[:120]}", flush=True)
        state["code_execution_result"]  = output
        state["code_execution_success"] = success
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 4c — Verify Result  (computational path)
    # FIX Issue 5: "notes" line matched with startswith, not substring
    # ══════════════════════════════════════════════════════════════════════════
    def verify_result(state: AgentState) -> AgentState:
        if not state.get("code_execution_success"):
            state["result_verification_passed"] = False
            state["result_verification_notes"]  = "Skipped: code execution failed."
            return state
        prompt = RESULT_VERIFICATION_PROMPT.format(
            question=state["question"],
            result=state["code_execution_result"],
        )
        raw      = llm_call(client, model, prompt, max_tokens=100).lower()
        notes    = "unknown"
        units_ok = mag_ok = complete = "yes"
        for line in raw.splitlines():
            if line.startswith("units_correct"):
                vp = line.split(":", 1)[-1].strip()
                units_ok = "no" if re.search(r'\bno\b', vp) else "yes"
            elif line.startswith("magnitude_plausible"):
                vp = line.split(":", 1)[-1].strip()
                mag_ok = "no" if re.search(r'\bno\b', vp) else "yes"
            elif line.startswith("answer_complete"):
                vp = line.split(":", 1)[-1].strip()
                complete = "no" if re.search(r'\bno\b', vp) else "yes"
            # FIX Issue 5: use startswith instead of substring ────────────────
            elif line.startswith("notes"):
                notes = line.split(":", 1)[-1].strip()
            # ──────────────────────────────────────────────────────────────────
        passed = (units_ok == "yes" and mag_ok == "yes" and complete == "yes")
        print(
            f"  🔬 Verification — units:{units_ok} magnitude:{mag_ok} "
            f"complete:{complete} → {'✅' if passed else '❌'}",
            flush=True,
        )
        state["result_verification_passed"] = passed
        state["result_verification_notes"]  = notes
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 4d — Format Code Answer  (computational path)
    # ══════════════════════════════════════════════════════════════════════════
    def format_code_answer(state: AgentState) -> AgentState:
        prompt = CODE_ANSWER_FORMAT_PROMPT.format(
            question=state["question"],
            execution_result=state.get("code_execution_result", ""),
            context_summary=state["rag_context"][:1000],
        )
        answer = llm_call(client, model, prompt, system=SYSTEM_PROMPT, max_tokens=600)
        print(f"  ✍️  Code answer formatted ({len(answer)} chars)", flush=True)
        state["answer"]         = answer
        state["low_confidence"] = not state.get("result_verification_passed", False)
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # ROUTING FUNCTIONS
    # ══════════════════════════════════════════════════════════════════════════

    def route_after_classification(
        state: AgentState,
    ) -> Literal["multimodal_triage"]:
        # Stage 0.5 now runs for ALL queries (computational/factual, simple/complex)
        # before entering Node 2 retrieval.
        return "multimodal_triage"

    def route_after_relevance(
        state: AgentState,
    ) -> Literal["rewrite_query", "generate_answer", "generate_code"]:
        verdict      = state["relevance_verdict"]
        max_attempts = MAX_RETRIEVAL_ATTEMPTS
        needs_retry  = verdict in ("no", "partial")
        under_limit  = state["retrieval_attempts"] < max_attempts
        if needs_retry and under_limit:
            return "rewrite_query"
        if state.get("query_type") == "computational":
            return "generate_code"
        return "generate_answer"

    def route_after_execution(
        state: AgentState,
    ) -> Literal["verify_result", "generate_code", "generate_answer"]:
        if state.get("code_execution_success"):
            return "verify_result"
        if state.get("code_generation_attempts", 0) < MAX_CODE_GEN_ATTEMPTS:
            return "generate_code"
        print("  ⚠️  Code execution failed — falling back to text generation", flush=True)
        return "generate_answer"

    def route_after_verification(
        state: AgentState,
    ) -> Literal["format_code_answer", "generate_code", "generate_answer"]:
        """
        FIX Issue 7: one code-regeneration retry before falling back to text.
        Pass → format answer.
        Fail + attempts < MAX_CODE_GEN_ATTEMPTS → regenerate code with error feedback.
        Fail + attempts exhausted → fall back to text generation.
        """
        if state.get("result_verification_passed"):
            return "format_code_answer"
        if state.get("code_generation_attempts", 0) < MAX_CODE_GEN_ATTEMPTS:
            print("  🔁 Verification failed — retrying code generation with error feedback",
                  flush=True)
            return "generate_code"
        print("  ⚠️  Verification failed after max retries — falling back to text generation",
              flush=True)
        return "generate_answer"

    def route_after_reflection(
        state: AgentState,
    ) -> Literal["refine_answer", "__end__"]:
        if not state["reflection_passed"] and state["reflection_cycles"] < MAX_REFLECTION_CYCLES:
            return "refine_answer"
        return "__end__"

    # ══════════════════════════════════════════════════════════════════════════
    # GRAPH CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════
    graph = StateGraph(AgentState)

    graph.add_node("classify_query_type", classify_query_type)
    graph.add_node("multimodal_triage",   multimodal_triage)     # Stage 0.5 (new)
    graph.add_node("retrieve_docs",       retrieve_docs)
    graph.add_node("check_relevance",     check_relevance)
    graph.add_node("rewrite_query",       rewrite_query)
    graph.add_node("generate_answer",     generate_answer)
    graph.add_node("reflect",             reflect)
    graph.add_node("refine_answer",       refine_answer)
    graph.add_node("generate_code",       generate_code)
    graph.add_node("execute_code",        execute_code)
    graph.add_node("verify_result",       verify_result)
    graph.add_node("format_code_answer",  format_code_answer)

    graph.set_entry_point("classify_query_type")

    # Stage 0.5 branch: all query types -> multimodal_triage -> retrieve_docs
    graph.add_conditional_edges(
        "classify_query_type",
        route_after_classification,
        {
            "multimodal_triage": "multimodal_triage",
        },
    )
    graph.add_edge("multimodal_triage", "retrieve_docs")

    graph.add_edge("retrieve_docs",  "check_relevance")

    graph.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "rewrite_query"  : "rewrite_query",
            "generate_answer": "generate_answer",
            "generate_code"  : "generate_code",
        },
    )

    graph.add_edge("rewrite_query", "retrieve_docs")   # retrieval loop

    # Factual path
    graph.add_edge("generate_answer", "reflect")
    graph.add_conditional_edges(
        "reflect",
        route_after_reflection,
        {"refine_answer": "refine_answer", "__end__": END},
    )
    graph.add_edge("refine_answer", "reflect")          # reflection loop

    # Computational path
    graph.add_edge("generate_code", "execute_code")
    graph.add_conditional_edges(
        "execute_code",
        route_after_execution,
        {
            "verify_result": "verify_result",
            "generate_code": "generate_code",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_conditional_edges(
        "verify_result",
        route_after_verification,
        {
            "format_code_answer": "format_code_answer",
            "generate_code"     : "generate_code",     # FIX Issue 7: retry edge
            "generate_answer"   : "generate_answer",
        },
    )
    graph.add_edge("format_code_answer", END)

    return graph.compile()
