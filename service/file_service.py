from enum import Enum
import os
import uuid
from datetime import datetime, timezone
from turtle import up
from typing import Any, Callable, Dict

from fastapi import File, HTTPException, UploadFile
from qdrant_client.http import models as qmodels

from service import chunkings
from service.chunkings import embed_query, semantic_chunker, sliding_window_chunker
from service.llm_service import LlmService
from service.models import UserDocs
from service.parsers import Parsers


def safe_supabase_database_action(action: Callable[[], Any]) -> Dict[str, Any]:
    try:
        response = action()
        data = getattr(response, "data", None)
        error = getattr(response, "error", None)
        if not data and hasattr(response, "__dict__"):
            data = response.__dict__.get("data")
        if error:
            raise HTTPException(status_code=500, detail=f"Supabase Error: {error}")
        if data is None:
            raise HTTPException(
                status_code=500, detail="No data returned from Supabase"
            )
        return {"success": True, "data": data, "error": None}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected Supabase error: {str(e)}"
        )


def safe_supabase_storage_action(action: Callable[[], Any]) -> Dict[str, Any]:
    try:
        response = action()
        if hasattr(response, "error") and response.error:
            raise HTTPException(
                status_code=500, detail=f"Supabase Storage Error: {response.error}"
            )
        if hasattr(response, "__dict__"):
            data = response.__dict__
        elif isinstance(response, dict):
            data = response
        else:
            data = {"result": str(response)}
        return {"success": True, "data": data, "error": None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected Supabase Storage error: {str(e)}"
        )


# semantic_chunks = semantic_chunker(
#     file_bytes=file_bytes,
#     max_chunk_size=1200,
#     mode="paragraph"
# )

# window_chunks = sliding_window_chunker(
#     file_bytes=file_bytes,
#     chunk_size=1000,
#     overlap=200
# )


def parse_uploaded_at(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def recency_score(uploaded_at_iso: str, now: datetime) -> float:
    uploaded_at = parse_uploaded_at(uploaded_at_iso)
    age_seconds = (now - uploaded_at).total_seconds()
    return 1.0 / (1.0 + age_seconds)


IMPORTANT_SECTIONS = {"definition", "definitions", "overview", "introduction"}


def adjacency_score(
    chunk_index: int,
    relevant_indices: set[int],
) -> float:
    if chunk_index - 1 in relevant_indices:
        return 0.5
    if chunk_index + 1 in relevant_indices:
        return 0.5
    return 0.0


def final_score(
    similarity: float,
    recency: float,
    hierarchy: float,
    adjacency: float,
    W1=0.55,
    W2=0.20,
    W3=0.15,
    W4=0.10,
) -> float:
    return similarity * W1 + recency * W2 + hierarchy * W3 + adjacency * W4


def hierarchy_score(section_path: list[str]) -> float:
    for section in section_path:
        if section.lower() in IMPORTANT_SECTIONS:
            return 1.0
    return 0.0


class Vectordb_Service:
    @staticmethod
    async def store_embeddings(
        embedded_chunks,
        vdb,
        filename: str,
    ):
        if not embedded_chunks:
            return {
                "data": "Uploaded doc is not parsable",
                "success": False,
            }

        points = []

        for item in embedded_chunks:
            chunk = item["chunk"]
            embedding = item["embedding"]

            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "filename": filename,
                        "document_id": chunk.metadata.document_id,
                        "page_start": chunk.metadata.page_start,
                        "page_end": chunk.metadata.page_end,
                        "section_path": chunk.metadata.section_path,
                        "chunk_index": chunk.metadata.chunk_index,
                        "uploaded_at": chunk.metadata.uploaded_at,
                        "text": chunk.text,
                    },
                )
            )

        vector_db_response = await vdb.upsert(
            collection_name="user_docs",
            points=points,
        )

        return {
            "data": vector_db_response,
            "success": True,
        }

    @staticmethod
    async def basic_semantic_search(
        query: str,
        vdb,
        top_k: int = 20,
        filename: str | None = None,
    ):

        query_vector = embed_query(query)
        search_filter = None
        if filename:
            search_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="filename",
                        match=qmodels.MatchValue(value=filename),
                    )
                ]
            )

        raw = await vdb.query_points(
            collection_name="user_docs",
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=search_filter,
        )

        if isinstance(raw, tuple):
            points = raw[1] if len(raw) > 1 else []
        elif hasattr(raw, "points"):
            points = raw.points
        else:
            points = raw
        results = []

        if not points:
            return []

        for point in points:
            try:
                results.append(
                    {
                        "score": getattr(point, "score", 0),
                        "text": point.payload.get("text") if point.payload else None,
                        "document_id": (
                            point.payload.get("document_id") if point.payload else None
                        ),
                        "filename": (
                            point.payload.get("filename") if point.payload else None
                        ),
                        "page_start": (
                            point.payload.get("page_start") if point.payload else None
                        ),
                        "page_end": (
                            point.payload.get("page_end") if point.payload else None
                        ),
                        "section_path": (
                            point.payload.get("section_path") if point.payload else None
                        ),
                        "chunk_index": (
                            point.payload.get("chunk_index") if point.payload else None
                        ),
                        "uploaded_at": (
                            point.payload.get("uploaded_at") if point.payload else None
                        ),
                    }
                )
            except (AttributeError, TypeError):
                results.append(
                    {
                        "score": (
                            point.get("score", 0) if isinstance(point, dict) else 0
                        ),
                        "text": (
                            point.get("payload", {}).get("text")
                            if isinstance(point, dict)
                            else None
                        ),
                    }
                )

        return await Vectordb_Service.rerank_chunks(
            retrieved_chunks=results,
            now=datetime.now(timezone.utc),
        )

    @staticmethod
    async def rerank_chunks(
        retrieved_chunks: list[dict],
        now: datetime | None = None,
    ) -> list[dict]:
        if not now:
            now = datetime.now(timezone.utc)

        top_similar = sorted(
            retrieved_chunks,
            key=lambda x: x["score"],
            reverse=True,
        )[:5]

        relevant_indices = {c["chunk_index"] for c in top_similar}

        reranked = []

        for chunk in retrieved_chunks:
            sim = chunk["score"]
            rec = recency_score(chunk["uploaded_at"], now)
            hier = hierarchy_score(chunk["section_path"])
            adj = adjacency_score(chunk["chunk_index"], relevant_indices)

            score = final_score(
                similarity=sim,
                recency=rec,
                hierarchy=hier,
                adjacency=adj,
            )

            reranked.append(
                {
                    **chunk,
                    "final_score": score,
                }
            )

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked


class ChunkingMethod(Enum):
    SLIDING_WINDOW = "SLIDING_WINDOW"
    SEMANTIC_CHUNKING = "SEMANTIC_CHUNKING"


class File_Service:

    @staticmethod
    async def upload_file_info_in_db(data: dict, db):
        try:
            async with db.begin():
                db_data = UserDocs(**data)
                db.add(db_data)
                await db.flush()
            return {
                "data": "File data in DB",
                "success": True,
            }

        except Exception as e:
            await db.rollback()
            raise e

    @staticmethod
    async def upload_file_info_in_store(file: UploadFile, file_info, store):
        try:
            bucket_name = "eval_user_docs"
            await store.storage.from_(bucket_name).upload(
                path=file_info["id"],
                file=await file.read(),
                file_options={
                    "content-type": file.content_type,
                    "upsert": False,
                },
            )

            return {
                "success": True,
                "path": f"{bucket_name}/{file_info["id"]}",
            }

        except Exception as e:
            raise ValueError(str(e))

    @staticmethod
    async def get_uploaded_file_info(file: UploadFile) -> dict:
        mimetype = file.content_type if file.content_type else ""
        size = file.size if file.size else 0
        filename = file.filename if file.filename else ""
        return {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "size_bytes": size,
            "size_kb": round(size / 1024, 2),
            "size_mb": round(size / (1024 * 1024), 2),
            "extension": filename.split(".")[-1],
            "mime_type": mimetype,
            "uploaded_at": str(datetime.now(timezone.utc)),
        }

    @staticmethod
    async def get_document_citations(query, vdb):
        return await Vectordb_Service.basic_semantic_search(
            query=query,
            vdb=vdb,
            top_k=5,
        )

    @staticmethod
    async def upload_single_file(
        db,
        vdb,
        store,
        chunking_method: str,
        file: UploadFile = File(...),
    ):

        file_info = await File_Service.get_uploaded_file_info(file)
        file_bytes = await file.read()

        await File_Service.upload_file_info_in_db(
            data=file_info,
            db=db,
        )

        await File_Service.upload_file_info_in_store(
            file=file,
            file_info=file_info,
            store=store,
        )

        if chunking_method == ChunkingMethod.SEMANTIC_CHUNKING:
            chunks = semantic_chunker(
                document_id=file_info["id"],
                file_bytes=file_bytes,
                max_chunk_size=300,
                mode="paragraph",
            )
            await Vectordb_Service.store_embeddings(
                filename=file_info["filename"],
                embedded_chunks=chunks,
                vdb=vdb,
            )
        elif chunking_method == ChunkingMethod.SLIDING_WINDOW:
            chunks = sliding_window_chunker(
                chunk_size=100,
                document_id=file_info["id"],
                file_bytes=file_bytes,
                overlap=80,
            )
            await Vectordb_Service.store_embeddings(
                filename=file_info["filename"],
                embedded_chunks=chunks,
                vdb=vdb,
            )

        return {
            "data": "Info stored DB, File storage and Vector DB along with embeddings",
            "success": True,
        }

    @staticmethod
    async def get_output_from_llm(query, vdb):
        reranked_list = await Vectordb_Service.basic_semantic_search(
            query=query,
            vdb=vdb,
            top_k=5,
        )
        return await LlmService.get_structured_reranked_output(
            reranked_data=reranked_list,
            query=query,
        )
