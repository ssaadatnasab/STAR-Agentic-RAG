"""Hybrid retrieval and corpus loading (BM25 + dense + MMR + reranking)."""
import gc
import hashlib
import importlib
import json
import os
import pickle
import re
from pathlib import Path
import numpy as np

def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]

def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0] if b.ndim > 1 else 0))
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_safe = np.divide(a, np.where(a_norm == 0, 1.0, a_norm))
    b_safe = np.divide(b, np.where(b_norm == 0, 1.0, b_norm))
    return np.clip(a_safe @ b_safe.T, 0.0, 1.0)

def _tokenize_for_bm25(text: str) -> list[str]:
    return [t for t in str(text).lower().split() if t]

def _compute_corpus_signature(jsonl_files: list[Path]) -> str:
    hasher = hashlib.sha256()
    for fp in sorted(jsonl_files, key=lambda p: p.name):
        stat = fp.stat()
        hasher.update(f"{fp.name}|{stat.st_size}|{int(stat.st_mtime)}".encode())
    return hasher.hexdigest()

def _extract_jsonl_text(obj: dict) -> str:
    def _stringify_value(val) -> str:
        if isinstance(val, (str, int, float)):
            return str(val).strip()
        if isinstance(val, dict):
            parts = []
            for k, v in val.items():
                ks = str(k).strip()
                vs = _stringify_value(v)
                if ks and vs:
                    parts.append(f"{ks}: {vs}")
                elif ks and v is None:
                    parts.append(ks)
                elif vs:
                    parts.append(vs)
            return "\n".join(p for p in parts if p).strip()
        if isinstance(val, (list, tuple, set)):
            parts = [_stringify_value(v) for v in val]
            return "\n".join(p for p in parts if p).strip()
        return ""

    preferred_keys = ["text", "content", "chunk", "markdown", "body", "input", "output"]
    parts = []
    for key in preferred_keys:
        s = _stringify_value(obj.get(key))
        if s:
            parts.append(s)
    if parts:
        return "\n".join(parts)

    if len(obj) == 1:
        only_key, only_val = next(iter(obj.items()))
        s = _stringify_value(only_val)
        if s:
            return s
        if only_val is None:
            return str(only_key).strip()

    for k, val in obj.items():
        s = _stringify_value(val)
        if s:
            return s
        if val is None:
            ks = str(k).strip()
            if ks:
                return ks
    return ""

def _extract_raw_line_text(line: str) -> str:
    s = str(line).strip()
    if not s:
        return ""
    if not any(ch.isalnum() for ch in s):
        return ""
    return s

def _mmr_select(docs, relevance_scores, doc_embeddings, lambda_mult, target_k):
    if not docs:
        return []
    target_k = max(1, min(target_k, len(docs)))
    selected, remaining = [], list(range(len(docs)))
    while remaining and len(selected) < target_k:
        if not selected:
            best = max(remaining, key=lambda i: relevance_scores[i])
            selected.append(best)
            remaining.remove(best)
            continue
        best_idx, best_val = None, -1e9
        for idx in remaining:
            sel_emb  = doc_embeddings[selected]
            cand_emb = doc_embeddings[idx:idx + 1]
            max_sim  = float(
                _cosine_similarity_matrix(cand_emb, sel_emb).max()
            ) if sel_emb.size else 0.0
            score = lambda_mult * relevance_scores[idx] - (1 - lambda_mult) * max_sim
            if score > best_val:
                best_val, best_idx = score, idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected

def _is_parent_child_corpus(corpus_dir: Path) -> bool:
    return (
        (corpus_dir / "strategy3_children.jsonl").exists()
        and (corpus_dir / "strategy3_parents.jsonl").exists()
    )

def _load_parent_child_corpus(corpus_dir: Path, Document):
    children_path = corpus_dir / "strategy3_children.jsonl"
    parents_path = corpus_dir / "strategy3_parents.jsonl"

    parent_lookup: dict[str, object] = {}
    with open(parents_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️  Skipping invalid parent JSON: {parents_path.name}:{line_num}", flush=True)
                continue

            chunk_id = obj.get("chunk_id")
            text = obj.get("text", "").strip()
            if not chunk_id or not text:
                continue

            parent_lookup[chunk_id] = Document(
                page_content=text,
                metadata={
                    "chunk_id": chunk_id,
                    "source": obj.get("source_file", parents_path.name),
                    "section_heading": obj.get("section_heading", ""),
                    "chunk_role": "parent",
                    "line": line_num,
                },
            )

    child_docs = []
    with open(children_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️  Skipping invalid child JSON: {children_path.name}:{line_num}", flush=True)
                continue

            text = obj.get("text", "").strip()
            parent_id = obj.get("parent_id")
            if not text or not parent_id:
                continue

            child_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "chunk_id": obj.get("chunk_id"),
                        "parent_id": parent_id,
                        "source": obj.get("source_file", children_path.name),
                        "section_heading": obj.get("section_heading", ""),
                        "chunk_role": "child",
                        "line": line_num,
                        "chunk_index": len(child_docs),
                    },
                )
            )

    if not child_docs:
        raise ValueError(f"No child chunks loaded from {children_path}")
    if not parent_lookup:
        raise ValueError(f"No parent chunks loaded from {parents_path}")

    missing_parents = {
        doc.metadata["parent_id"]
        for doc in child_docs
        if doc.metadata["parent_id"] not in parent_lookup
    }
    if missing_parents:
        print(
            f"⚠️  {len(missing_parents)} child chunks reference missing parents; "
            "those children will be skipped at retrieval time.",
            flush=True,
        )

    return child_docs, parent_lookup

def build_hybrid_retriever(
    corpus_path, k, embedding_model,
    reranker_model, bm25_top_k, dense_top_k, fused_top_k,
    mmr_lambda, mmr_target_k,
    parent_child: bool | None = None,
):
    Document = importlib.import_module("langchain_core.documents").Document
    try:
        HuggingFaceEmbeddings = importlib.import_module(
            "langchain_huggingface"
        ).HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = importlib.import_module(
            "langchain_community.embeddings"
        ).HuggingFaceEmbeddings
    CrossEncoder = importlib.import_module("sentence_transformers").CrossEncoder

    corpus_dir = Path(corpus_path)
    use_parent_child = _is_parent_child_corpus(corpus_dir) if parent_child is None else parent_child
    parent_lookup: dict[str, object] | None = None

    if use_parent_child:
        if not _is_parent_child_corpus(corpus_dir):
            raise ValueError(
                f"Parent-child corpus not found in {corpus_path}. "
                "Expected strategy3_children.jsonl and strategy3_parents.jsonl."
            )
        docs, parent_lookup = _load_parent_child_corpus(corpus_dir, Document)
        corpus_signature = _compute_corpus_signature([corpus_dir / "strategy3_children.jsonl"])
        print(
            f"📚 Loaded parent-child corpus: {len(docs)} child chunks for retrieval, "
            f"{len(parent_lookup)} parent chunks for LLM context.",
            flush=True,
        )
    else:
        jsonl_files = sorted(corpus_dir.glob("*.jsonl"))
        markdown_files = sorted(corpus_dir.rglob("*.md"))
        docs: list = []

        if jsonl_files:
            corpus_signature = _compute_corpus_signature(jsonl_files)
            print(f"📚 Loading corpus from {len(jsonl_files)} JSONL files ({corpus_dir})...", flush=True)
            for jf in jsonl_files:
                try:
                    chunk_idx = 0
                    with open(jf, encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                            try:
                                obj  = json.loads(line)
                                text = _extract_jsonl_text(obj)
                                if text:
                                    docs.append(Document(
                                        page_content=text,
                                        metadata={
                                            "source"      : jf.name,
                                            "line"        : line_num,
                                            "chunk_index" : chunk_idx,
                                        },
                                    ))
                                    chunk_idx += 1
                            except json.JSONDecodeError:
                                text = _extract_raw_line_text(line)
                                if text:
                                    docs.append(Document(
                                        page_content=text,
                                        metadata={
                                            "source"      : jf.name,
                                            "line"        : line_num,
                                            "chunk_index" : chunk_idx,
                                            "parse"       : "raw_line",
                                        },
                                    ))
                                    chunk_idx += 1
                except Exception as e:
                    print(f"⚠️  Failed to read {jf.name}: {e}", flush=True)
        elif markdown_files:
            corpus_signature = _compute_corpus_signature(markdown_files)
            print(f"📚 Loading corpus from {len(markdown_files)} markdown files ({corpus_dir})...", flush=True)
            for md_index, markdown_file in enumerate(markdown_files):
                try:
                    text = markdown_file.read_text(encoding="utf-8").strip()
                    if text:
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"source": markdown_file.name, "chunk_index": md_index},
                            )
                        )
                except Exception as e:
                    print(f"⚠️  Failed to read markdown file {markdown_file.name}: {e}", flush=True)
        else:
            raise ValueError(f"No .jsonl or .md files found in {corpus_path}")

    if not docs:
        raise ValueError(f"No documents loaded from corpus: {corpus_path}")
    print(f"✅ Loaded {len(docs)} documents from {corpus_dir}", flush=True)

    tokenized_corpus = [_tokenize_for_bm25(d.page_content) for d in docs]
    # MEMORY OPT: Force embeddings to CPU with memory-efficient settings
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={
            "device": "cpu",
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 8,  # Reduced batch size to 8 (from default 32)
        },
    )

    cache_prefix = "children_" if use_parent_child else ""
    cache_path      = corpus_dir / f"{cache_prefix}embeddings_cache.pkl"
    cache_meta_path = corpus_dir / f"{cache_prefix}embeddings_cache_meta.json"

    cache_loaded = False
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                doc_embedding_matrix = np.asarray(pickle.load(f), dtype=float)
            if doc_embedding_matrix.ndim != 2:
                raise ValueError(f"invalid embedding matrix ndim={doc_embedding_matrix.ndim}")
            cache_loaded = True
            print(f"✅ Loaded cached embeddings ({corpus_dir}).", flush=True)
        except Exception as e:
            print(f"⚠️  Cache load error ({e}), recomputing... ({corpus_dir})", flush=True)

    if not cache_loaded:
        print(f"⚙️  Computing embeddings (CPU, batched) for {corpus_dir}...", flush=True)
        # MEMORY OPT: Process embeddings in small batches to avoid GPU memory spike
        import gc
        batch_size = 16
        all_embeds = []
        for i in range(0, len(docs), batch_size):
            batch_docs = [d.page_content for d in docs[i:i+batch_size]]
            batch_embeds = embeddings.embed_documents(batch_docs)
            all_embeds.extend(batch_embeds)
            gc.collect()  # Force garbage collection after each batch
        doc_embedding_matrix = np.asarray(all_embeds, dtype=float)
        with open(cache_path, "wb") as f:
            pickle.dump(doc_embedding_matrix, f)
        with open(cache_meta_path, "w") as f:
            json.dump({
                "embedding_model": embedding_model,
                "rows": int(doc_embedding_matrix.shape[0]),
                "cols": int(doc_embedding_matrix.shape[1]) if doc_embedding_matrix.ndim == 2 else 0,
                "corpus_signature": corpus_signature,
            }, f, indent=2)
        print(f"✅ Embeddings computed and cached ({len(docs)} chunks, {corpus_dir}).", flush=True)

    if doc_embedding_matrix.shape[0] != len(docs):
        shared_n = min(int(doc_embedding_matrix.shape[0]), len(docs))
        print(
            f"⚠️  Cache/doc mismatch ({corpus_dir}): embeddings={doc_embedding_matrix.shape[0]} "
            f"docs={len(docs)}. Using first {shared_n} entries.",
            flush=True,
        )
        if shared_n <= 0:
            raise ValueError(f"No usable embeddings/docs overlap for {corpus_dir}")
        docs = docs[:shared_n]
        tokenized_corpus = tokenized_corpus[:shared_n]
        doc_embedding_matrix = doc_embedding_matrix[:shared_n]

    # Build or load FAISS index for fast dense search (inner product on L2-normalised vecs == cosine)
    use_faiss   = False
    faiss_index = None
    try:
        import faiss
        dim        = int(doc_embedding_matrix.shape[1])
        index_path = corpus_dir / f"{cache_prefix}faiss_index.idx"
        if index_path.exists():
            try:
                idx = faiss.read_index(str(index_path))
                if getattr(idx, "ntotal", -1) == doc_embedding_matrix.shape[0]:
                    faiss_index = idx
                    use_faiss   = True
                    print(f"✅ Loaded FAISS index from {index_path}", flush=True)
                else:
                    print("⚠️  FAISS index size mismatch — rebuilding.", flush=True)
            except Exception:
                print("⚠️  Failed to read FAISS index file — rebuilding.", flush=True)
        if not use_faiss:
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(doc_embedding_matrix.astype("float32"))
            try:
                faiss.write_index(faiss_index, str(index_path))
                print(f"✅ Built and saved FAISS IndexFlatIP (dim={dim}) to {index_path}", flush=True)
            except Exception:
                print("⚠️  Built FAISS index but failed to save to disk.", flush=True)
            use_faiss = True
    except Exception:
        print("⚠️  FAISS not available or failed — falling back to cosine search.", flush=True)

    bm25_model  = importlib.import_module("rank_bm25").BM25Okapi(tokenized_corpus)

    # Keep reranker on CPU explicitly to avoid CUDA/driver issues.
    reranker = CrossEncoder(reranker_model, device="cpu")
    if hasattr(reranker, "model"):
        reranker.model.eval()

    query_embed = embeddings.embed_query

    def retrieve(query: str) -> list:
        q_tokens    = _tokenize_for_bm25(query)
        bm25_scores = list(map(float, bm25_model.get_scores(q_tokens)))
        bm25_norm   = _min_max_normalize(bm25_scores)
        q_vec       = np.asarray(query_embed(query), dtype=float)

        # Dense search: FAISS when available, else brute-force cosine
        if use_faiss and faiss_index is not None:
            try:
                qvec = q_vec.reshape(1, -1).astype("float32")
                distances, topk = faiss_index.search(qvec, max(1, dense_top_k))
                dense_top_indices = topk.reshape(-1).tolist()
                dense_scores_arr  = np.zeros(len(docs), dtype=float)
                for idx_pos, sc in zip(dense_top_indices, distances.reshape(-1)):
                    dense_scores_arr[int(idx_pos)] = float(np.clip(sc, 0.0, 1.0))
                dense_scores = dense_scores_arr.tolist()
            except Exception:
                dense_scores = _cosine_similarity_matrix(
                    doc_embedding_matrix, q_vec.reshape(1, -1)
                ).reshape(-1).tolist()
        else:
            dense_scores = _cosine_similarity_matrix(
                doc_embedding_matrix, q_vec.reshape(1, -1)
            ).reshape(-1).tolist()
        dense_norm   = _min_max_normalize(dense_scores)

        bm25_top   = np.argsort(bm25_scores)[::-1][:max(1, bm25_top_k)]
        dense_top  = np.argsort(dense_scores)[::-1][:max(1, dense_top_k)]
        candidates = list(dict.fromkeys(list(bm25_top) + list(dense_top)))

        # weighted fusion (BM25 0.4 / dense 0.6)
        fused = sorted(
            [(idx, float(np.clip(0.4 * bm25_norm[idx] + 0.6 * dense_norm[idx], 0, 1)))
             for idx in candidates],
            key=lambda x: x[1], reverse=True
        )[:max(1, fused_top_k)]

        fused_idx = [i for i, _ in fused]
        fused_rel = [s for _, s in fused]
        fused_emb = doc_embedding_matrix[fused_idx]

        mmr_sel  = _mmr_select(
            [docs[i] for i in fused_idx], fused_rel, fused_emb,
            lambda_mult=mmr_lambda, target_k=max(1, mmr_target_k),
        )
        div_idx  = [fused_idx[i] for i in mmr_sel]
        div_docs = [docs[i] for i in div_idx]
        print(f"    MMR ({corpus_dir.name}): {len(fused_idx)} fused → {len(div_docs)} diverse → reranker → top {k}",
              flush=True)

        pairs     = [[query, d.page_content] for d in div_docs]
        try:
            if pairs:
                try:
                    rr_raw = reranker.predict(
                        pairs,
                        batch_size=16,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                except TypeError:
                    rr_raw = reranker.predict(pairs)
                rr_scores = rr_raw.tolist() if hasattr(rr_raw, "tolist") else list(rr_raw)
            else:
                rr_scores = []
        except Exception as e:
            print(f"⚠️  Reranking error ({type(e).__name__}): {repr(e)}", flush=True)
            rr_scores = [0.0] * len(pairs)
        reranked  = sorted(
            list(zip(div_idx, div_docs, rr_scores)), key=lambda x: x[2], reverse=True
        )[:max(1, k)]

        if use_parent_child and parent_lookup is not None:
            final = []
            seen_parent_ids: set[str] = set()
            for doc_idx, child_doc, score in reranked:
                doc_idx = int(doc_idx)
                parent_id = child_doc.metadata.get("parent_id")
                if not parent_id or parent_id in seen_parent_ids:
                    continue
                parent_doc = parent_lookup.get(parent_id)
                if parent_doc is None:
                    continue

                seen_parent_ids.add(parent_id)
                resolved_doc = Document(
                    page_content=parent_doc.page_content,
                    metadata=dict(parent_doc.metadata),
                )
                resolved_doc.metadata["bm25_score"] = float(bm25_scores[doc_idx])
                resolved_doc.metadata["dense_score"] = float(dense_scores[doc_idx])
                resolved_doc.metadata["fused_score"] = float(0.4 * bm25_norm[doc_idx] + 0.6 * dense_norm[doc_idx])
                resolved_doc.metadata["rerank_score"] = float(score)
                resolved_doc.metadata["matched_child_id"] = child_doc.metadata.get("chunk_id")
                resolved_doc.metadata["matched_child_text"] = child_doc.page_content[:240]
                final.append(resolved_doc)
                if len(final) >= max(1, k):
                    break
            return final

        final = []
        for doc_idx, doc, score in reranked:
            doc_idx = int(doc_idx)  # Ensure doc_idx is an integer
            doc.metadata = dict(doc.metadata)
            doc.metadata["bm25_score"]   = float(bm25_scores[doc_idx])
            doc.metadata["dense_score"]  = float(dense_scores[doc_idx])
            doc.metadata["fused_score"]  = float(
                0.4 * bm25_norm[doc_idx] + 0.6 * dense_norm[doc_idx]
            )
            doc.metadata["rerank_score"] = float(score)
            final.append(doc)
        return final

    return retrieve

def docs_to_context(docs: list, max_chars: int = 8000) -> str:
    segments, total = [], 0
    for i, doc in enumerate(docs, 1):
        src   = doc.metadata.get("source", "?")
        score = doc.metadata.get("rerank_score", None)
        tag   = f" | rerank={score:.4f}" if isinstance(score, (int, float)) else ""
        matched_child_id = doc.metadata.get("matched_child_id")
        child_tag = f" | matched_child={matched_child_id}" if matched_child_id else ""
        seg   = f"[{i}] From {src}{tag}{child_tag}\n{doc.page_content}\n"
        remaining = max_chars - total
        if len(seg) > remaining:
            if not segments and remaining > 0:
                segments.append(seg[:remaining])
            print(f"⚠️  Context truncated at chunk {i}/{len(docs)}", flush=True)
            break
        segments.append(seg)
        total += len(seg)
    return "\n".join(segments).strip()

def _build_retriever_safe(label: str, corpus_path: str, k: int, embedding_model: str,
                           reranker_model: str, bm25_top_k: int, dense_top_k: int,
                           fused_top_k: int, mmr_lambda: float, mmr_target_k: int):
    """
    Build a hybrid retriever for a Stage 0.5 corpus. Returns None (with a loud
    warning) instead of crashing the whole pipeline if the corpus folder is
    missing or fails to build — silent failures are more dangerous than loud ones.
    """
    corpus_dir = Path(corpus_path)
    if not corpus_dir.exists():
        print(f"⚠️  Stage 0.5 — {label} corpus path does not exist: {corpus_path}. "
              f"This corpus will be treated as UNAVAILABLE.", flush=True)
        return None

    jsonl_files = sorted(corpus_dir.glob("*.jsonl"))
    markdown_files = sorted(corpus_dir.rglob("*.md"))
    if not jsonl_files and not markdown_files:
        print(f"ℹ️  Stage 0.5 — {label} corpus has no .jsonl/.md files. "
              f"Treating as UNAVAILABLE.", flush=True)
        return None

    if jsonl_files:
        has_non_empty_text = False
        try:
            for jf in jsonl_files:
                with open(jf, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            text = _extract_jsonl_text(obj)
                        except json.JSONDecodeError:
                            text = _extract_raw_line_text(line)
                        if text:
                            has_non_empty_text = True
                            break
                if has_non_empty_text:
                    break
        except Exception as e:
            print(f"⚠️  Stage 0.5 — could not validate {label} corpus ({e}). "
                  f"Treating as UNAVAILABLE.", flush=True)
            return None
        if not has_non_empty_text:
            print(f"ℹ️  Stage 0.5 — {label} corpus JSONL has no non-empty text entries. "
                  f"Treating as UNAVAILABLE.", flush=True)
            return None

    try:
        return build_hybrid_retriever(
            corpus_path=corpus_path, k=k, embedding_model=embedding_model,
            reranker_model=reranker_model, bm25_top_k=bm25_top_k, dense_top_k=dense_top_k,
            fused_top_k=fused_top_k, mmr_lambda=mmr_lambda, mmr_target_k=mmr_target_k,
            parent_child=(label == "parent_child"),
        )
    except Exception as e:
        print(f"⚠️  Stage 0.5 — failed to build {label} retriever ({e}). "
              f"This corpus will be treated as UNAVAILABLE.", flush=True)
        return None
