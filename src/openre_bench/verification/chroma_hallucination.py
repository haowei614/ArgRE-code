"""ChromaDB + OpenAI ``text-embedding-ada-002`` hallucination screen (Phase 4 Layer 2)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def _corpus_fingerprint(corpus_dir: Path) -> str:
    digest = hashlib.sha256()
    if not corpus_dir.exists():
        return ""
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(corpus_dir).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            return ""
    return digest.hexdigest()


def _iter_corpus_documents(corpus_dir: Path, *, max_chunks: int = 4000) -> list[str]:
    """Flatten knowledge corpus into chunk strings (aligned with Phase~1 RAG file types)."""

    if not corpus_dir.exists():
        return []
    allowed = {".md", ".txt", ".json", ".py"}
    chunks: list[str] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        compact = " ".join(text.split())
        if len(compact) < 40:
            continue
        step = 800
        if len(compact) > step:
            for start in range(0, len(compact), step):
                piece = compact[start : start + step].strip()
                if len(piece) >= 80:
                    chunks.append(piece)
        else:
            chunks.append(compact)
        if len(chunks) >= max_chunks:
            break
    return chunks


def chroma_hallucination_pass(
    *,
    gsn_elements: list[dict[str, Any]],
    corpus_dir: Path,
    persist_root: Path,
    openai_api_key: str,
    tau_h: float = 0.60,
) -> dict[str, Any]:
    """For each goal description, nearest-neighbor cosine vs Chroma index; flag if similarity < tau_h."""

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except Exception as exc:  # pragma: no cover
        return {
            "enabled": True,
            "skipped": True,
            "skip_reason": f"chromadb_import_failed: {exc}",
            "embedding_model": "text-embedding-ada-002",
            "tau_h": tau_h,
            "per_goal": [],
        }

    fingerprint = _corpus_fingerprint(corpus_dir)
    if not fingerprint:
        return {
            "enabled": True,
            "skipped": True,
            "skip_reason": "empty_or_unreadable_corpus",
            "embedding_model": "text-embedding-ada-002",
            "tau_h": tau_h,
            "per_goal": [],
        }

    persist_root.mkdir(parents=True, exist_ok=True)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-ada-002",
    )
    client = chromadb.PersistentClient(path=str(persist_root))
    collection_name = f"standards_{fingerprint[:24]}"
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
    )

    if collection.count() == 0:
        docs = _iter_corpus_documents(corpus_dir)
        if not docs:
            return {
                "enabled": True,
                "skipped": True,
                "skip_reason": "no_corpus_chunks",
                "embedding_model": "text-embedding-ada-002",
                "tau_h": tau_h,
                "per_goal": [],
            }
        ids = [f"c{i}" for i in range(len(docs))]
        batch = 64
        for start in range(0, len(docs), batch):
            batch_docs = docs[start : start + batch]
            batch_ids = ids[start : start + batch]
            collection.add(ids=batch_ids, documents=batch_docs)

    per_goal: list[dict[str, Any]] = []
    for item in gsn_elements:
        if not isinstance(item, dict):
            continue
        desc = f"{item.get('name', '')}. {item.get('description', '')}".strip()
        eid = str(item.get("id", "")).strip() or "unknown"
        if len(desc) < 8:
            per_goal.append(
                {
                    "element_id": eid,
                    "best_similarity": None,
                    "flagged_low_support": False,
                    "note": "description_too_short",
                }
            )
            continue
        result = collection.query(query_texts=[desc[:4000]], n_results=1)
        distances = result.get("distances") or []
        best_sim: float | None = None
        chunk_preview = ""
        if distances and distances[0]:
            d0 = float(distances[0][0])
            # Chroma cosine space: distance is 1 - cosine similarity for normalized embeddings.
            best_sim = float(np.clip(1.0 - d0, -1.0, 1.0))
        docs_out = result.get("documents") or []
        if docs_out and docs_out[0]:
            chunk_preview = str(docs_out[0][0])[:240]

        flagged = best_sim is not None and best_sim < tau_h
        per_goal.append(
            {
                "element_id": eid,
                "best_similarity": best_sim,
                "nearest_chunk_preview": chunk_preview,
                "flagged_low_support": flagged,
                "tau_h": tau_h,
            }
        )

    return {
        "enabled": True,
        "skipped": False,
        "corpus_dir": str(corpus_dir.resolve()),
        "corpus_fingerprint": fingerprint,
        "collection": collection_name,
        "persist_root": str(persist_root.resolve()),
        "embedding_model": "text-embedding-ada-002",
        "tau_h": tau_h,
        "per_goal": per_goal,
    }
