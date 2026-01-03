import os

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from supabase import AsyncClient, AsyncClientOptions, create_async_client

load_dotenv()

_storage: AsyncClient | None = None
_vdb = None

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "")


async def vector_database():
    global _vdb

    _vdb = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
        prefer_grpc=False,
    )

    collections_response = await _vdb.get_collections()
    collections = collections_response.collections
    collection_names = {c.name for c in collections}

    if "user_docs" not in collection_names:
        await _vdb.create_collection(
            collection_name="user_docs",
            vectors_config=qmodels.VectorParams(
                size=384,
                distance=qmodels.Distance.COSINE,
            ),
            optimizers_config=qmodels.OptimizersConfigDiff(default_segment_number=2),
            hnsw_config=qmodels.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )

        await _vdb.create_payload_index(
            collection_name="user_docs",
            field_name="file_name",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )

        print("✅ Qdrant collection created.")
    else:
        print("ℹ️ Qdrant collection already exists.")

    return _vdb


async def storage():
    global _storage
    _storage = await create_async_client(
        supabase_url=SUPABASE_URL,
        supabase_key=SUPABASE_SERVICE_ROLE_KEY,
        options=AsyncClientOptions(
            schema="public",
            auto_refresh_token=False,
            persist_session=False,
        ),
    )

    return _storage
