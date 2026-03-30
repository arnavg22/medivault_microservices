"""
vector_context.py

Queries the MongoDB Atlas Vector Search index to retrieve relevant clinical
document chunks for a specific patient. These chunks come from the RAG bot's
ingestion pipeline and may contain:
  - Prescription details and drug instructions
  - Lab report narratives and interpretations
  - Discharge summaries
  - Doctor's progress notes
  - Vital signs trends
  - Clinical chart annotations

The retrieved chunks are used to ENRICH the diet plan prompt with deeper
clinical context beyond what the structured MediVault API endpoints return.

This gives the diet bot access to narrative medical text — for example,
a discharge note saying "patient reports difficulty tolerating high-fiber
foods" which would never appear in a structured field.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from app.config.settings import Settings

logger = structlog.get_logger("vector_context")

# Shared async HF Inference client (reused across requests)
_hf_client = None


async def get_embedding(text: str, settings: Settings) -> List[float]:
    """
    Generate a vector embedding for the given text.

    Priority order:
      1. OpenAI (text-embedding-3-small, dim=768 to match RAG index)
      2. HuggingFace Inference API (BAAI/bge-base-en-v1.5, 768 dim, zero local RAM)
         (Gemini skipped: gemini-embedding-001 returns 3072 dims, mismatches Atlas index)

    Groq and Anthropic do not support embeddings — skipped.
    Never crashes — returns empty list on total failure.
    """
    # 1. Try OpenAI
    if settings.openai_api_key and settings.openai_api_key.strip():
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.openai_api_key)
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=768,  # Must match RAG bot's vector index (768)
            )
            embedding = response.data[0].embedding
            logger.info(
                "embedding_generated",
                provider="openai",
                model="text-embedding-3-small",
                dimensions=len(embedding),
            )
            return embedding
        except Exception as exc:
            logger.warning(
                "openai_embedding_failed",
                error=str(exc),
            )

    # 2. Gemini embedding skipped: gemini-embedding-001 returns 3072 dims but the
    #    Atlas vector index is fixed at 768 dims (matching the RAG bot's index).
    #    Enabling it would silently break vector search with a dimension mismatch.

    # 3. HuggingFace Inference API (free tier — same model, zero local RAM)
    if settings.hf_api_token and settings.hf_api_token.strip():
        try:
            embedding = await _get_hf_embedding(text, settings.embedding_model, settings.hf_api_token)
            logger.info(
                "embedding_generated",
                provider="huggingface-inference",
                model=settings.embedding_model,
                dimensions=len(embedding),
            )
            return embedding
        except Exception as exc:
            logger.error(
                "hf_inference_embedding_failed",
                model=settings.embedding_model,
                error=str(exc),
            )

    # Total failure — return empty
    logger.error("all_embedding_providers_failed")
    return []


async def _get_hf_embedding(text: str, model_name: str, api_token: str) -> List[float]:
    """
    Call the HuggingFace Inference API to generate an embedding.
    Uses the official AsyncInferenceClient — same models as sentence-transformers,
    but runs on HF's infrastructure (zero local RAM).
    """
    global _hf_client

    if _hf_client is None:
        from huggingface_hub import AsyncInferenceClient
        _hf_client = AsyncInferenceClient(token=api_token)

    result = await _hf_client.feature_extraction(text, model=model_name)

    # result is a numpy ndarray (768,) — convert to list
    embedding = result.tolist()

    # Normalize (matches sentence-transformers normalize_embeddings=True)
    norm = sum(x * x for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x / norm for x in embedding]

    return embedding


async def fetch_vector_context(
    patient_id: str,
    query: str,
    db: Any,
    settings: Settings,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Perform an Atlas Vector Search query against the medical_vectors
    collection to retrieve relevant clinical document chunks for this patient.

    Args:
        patient_id: MediVault patient ObjectId string
        query: Semantic search query for embeddings
        db: Motor async database handle
        settings: App settings (has collection_name, index_name)
        top_k: Number of chunks to retrieve

    Returns:
        List of dicts with: text, source, score, metadata
        Empty list on any error (graceful degradation).
    """
    if db is None:
        logger.warning("vector_search_skipped", reason="database_not_available")
        return []

    # 1. Generate embedding for the query
    embedding = await get_embedding(query, settings)
    if not embedding:
        logger.warning(
            "vector_search_skipped",
            reason="embedding_generation_failed",
        )
        return []

    collection = db[settings.mongodb_collection_name]

    # 2. Try Atlas Vector Search with post-filter on metadata.patient_id
    #    The existing vector index has filter fields for metadata.section_type
    #    and metadata.filename, but NOT metadata.patient_id. We use $match
    #    as a post-filter stage instead.
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": settings.vector_index_name,
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": top_k * 20,
                    "limit": top_k * 5,
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
            {
                "$match": {
                    "metadata.patient_id": patient_id,
                }
            },
            {"$limit": top_k},
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "source": "$metadata.section_type",
                    "score": 1,
                    "metadata": 1,
                }
            },
        ]

        logger.info(
            "vector_search_executing",
            collection=settings.mongodb_collection_name,
            index=settings.vector_index_name,
            top_k=top_k,
        )

        raw_results = await collection.aggregate(pipeline).to_list(top_k)

    except Exception as exc:
        logger.error(
            "vector_search_failed",
            collection=settings.mongodb_collection_name,
            index=settings.vector_index_name,
            patient_id=patient_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return []

    if not raw_results:
        logger.info(
            "vector_search_no_results",
            collection=settings.mongodb_collection_name,
        )
        return []

    # 3. Filter out low-relevance chunks
    MIN_SCORE = getattr(settings, 'vector_min_score', 0.60)
    results: List[Dict[str, Any]] = []
    for doc in raw_results:
        score = doc.get("score", 0.0)
        if score < MIN_SCORE:
            continue
        results.append(
            {
                "text": doc.get("text", ""),
                "source": doc.get("source", "general"),
                "score": round(score, 4),
                "metadata": {
                    "date": _extract_date(doc.get("metadata", {})),
                    "section_type": doc.get("metadata", {}).get(
                        "section_type", "general"
                    ),
                    "report_type": doc.get("metadata", {}).get(
                        "report_type", ""
                    ),
                    "filename": doc.get("metadata", {}).get("filename", ""),
                },
            }
        )

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    logger.info(
        "vector_context_fetched",
        patient_id=patient_id,
        total_raw=len(raw_results),
        after_score_filter=len(results),
    )

    return results


def _extract_date(metadata: Dict[str, Any]) -> str:
    """Extract the most relevant date from metadata."""
    for key in ("report_date", "created_at", "ingestion_date"):
        val = metadata.get(key)
        if val:
            if isinstance(val, datetime):
                return val.strftime("%Y-%m-%d")
            if isinstance(val, str):
                return val[:10]  # ISO date first 10 chars
    return "unknown"


def format_vector_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved vector chunks into a readable block for injection
    into the LLM system prompt.

    Example output:

    [Source: Discharge Summary | Date: 2025-01-15 | Relevance: 0.94]
    Patient was advised to avoid high-sodium foods due to hypertension...

    [Source: Prescription | Date: 2025-02-01 | Relevance: 0.91]
    Metformin 500mg twice daily — take with meals to reduce GI side effects...
    """
    if not chunks:
        return "  No additional clinical documents found in records."

    lines: List[str] = []
    for chunk in chunks:
        source = _format_source_name(chunk.get("source", "general"))
        date = chunk.get("metadata", {}).get("date", "unknown")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "").strip()

        if not text:
            continue

        # Truncate very long chunks to keep prompt size manageable
        if len(text) > 500:
            text = text[:497] + "..."

        header = f"  [{source} | Date: {date} | Relevance: {score:.2f}]"
        lines.append(header)
        lines.append(f"  {text}")
        lines.append("")

    return "\n".join(lines).rstrip() if lines else "  No additional clinical documents found in records."


def _format_source_name(source: str) -> str:
    """Convert section_type slug to readable label."""
    mapping = {
        "medications": "Prescription",
        "diagnosis": "Diagnosis",
        "lab_results": "Lab Report",
        "vitals": "Vital Signs",
        "allergies": "Allergy Record",
        "symptoms": "Symptoms",
        "procedures": "Procedures",
        "doctor_notes": "Doctor's Notes",
        "follow_up": "Follow-Up Notes",
        "patient_info": "Patient Info",
        "chief_complaint": "Chief Complaint",
        "medical_history": "Medical History",
        "table": "Clinical Table",
        "general": "Clinical Document",
    }
    return mapping.get(source, source.replace("_", " ").title())
