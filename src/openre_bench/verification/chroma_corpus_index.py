"""Shared Chroma + OpenAI ada-002 corpus indexing for Phase~1 RAG and Phase~4 Layer~2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from openre_bench.verification.chroma_hallucination import _corpus_fingerprint
from openre_bench.verification.chroma_hallucination import _iter_corpus_documents


def ensure_chroma_embedded_corpus(
    *,
    corpus_dir: Path,
    persist_root: Path,
    openai_api_key: str,
    namespace: str,
) -> dict[str, Any]:
    """Create or open a Chroma collection with ada-002 embeddings over ``corpus_dir``."""

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except Exception as exc:  # pragma: no cover
        return {
            "skipped": True,
            "skip_reason": f"chromadb_import_failed: {exc}",
            "collection_name": "",
            "persist_path": str(persist_root.resolve()),
            "chunk_count": 0,
        }

    fingerprint = _corpus_fingerprint(corpus_dir)
    if not fingerprint:
        return {
            "skipped": True,
            "skip_reason": "empty_or_unreadable_corpus",
            "collection_name": "",
            "persist_path": str(persist_root.resolve()),
            "chunk_count": 0,
        }

    persist_root.mkdir(parents=True, exist_ok=True)
    safe_ns = "".join(c if c.isalnum() else "_" for c in namespace)[:32]
    digest8 = hashlib.sha256(f"{fingerprint}:{namespace}".encode()).hexdigest()[:16]
    collection_name = f"{safe_ns}_{digest8}"

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-ada-002",
    )
    client = chromadb.PersistentClient(path=str(persist_root))
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
    )

    if collection.count() == 0:
        docs = _iter_corpus_documents(corpus_dir)
        if not docs:
            return {
                "skipped": True,
                "skip_reason": "no_corpus_chunks",
                "collection_name": collection_name,
                "persist_path": str(persist_root.resolve()),
                "chunk_count": 0,
            }
        ids = [f"c{i}" for i in range(len(docs))]
        batch = 64
        for start in range(0, len(docs), batch):
            collection.add(ids=ids[start : start + batch], documents=docs[start : start + batch])

    return {
        "skipped": False,
        "skip_reason": "",
        "collection_name": collection_name,
        "persist_path": str(persist_root.resolve()),
        "chunk_count": int(collection.count()),
        "corpus_fingerprint": fingerprint,
    }


def query_chroma_top_documents(
    *,
    query: str,
    persist_path: str,
    collection_name: str,
    openai_api_key: str,
    top_k: int = 3,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Return retrieved chunk dicts (id, document, similarity) for one query."""

    import numpy as np
    import chromadb
    from chromadb.utils import embedding_functions

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-ada-002",
    )
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_collection(name=collection_name, embedding_function=ef)
    result = collection.query(query_texts=[query[:4000]], n_results=max(1, int(top_k)))
    distances = result.get("distances") or []
    documents = result.get("documents") or []
    ids = result.get("ids") or []
    out: list[dict[str, Any]] = []
    sims: list[float] = []
    if not distances or not distances[0]:
        return out, sims
    row_d = distances[0]
    row_doc = documents[0] if documents and documents[0] else []
    row_ids = ids[0] if ids and ids[0] else []
    for idx, d0 in enumerate(row_d):
        sim = float(np.clip(1.0 - float(d0), -1.0, 1.0))
        sims.append(sim)
        doc_text = str(row_doc[idx]) if idx < len(row_doc) else ""
        cid = str(row_ids[idx]) if idx < len(row_ids) else f"hit_{idx}"
        out.append({"chunk_id": cid, "document": cid, "text": doc_text, "similarity": sim})
    return out, sims
