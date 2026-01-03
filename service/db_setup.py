import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from service.models import Base, UserDocs

load_dotenv()


SUPABASE_DB_URI = os.environ.get("SUPABASE_DB_URI", "")

engine = create_async_engine(
    SUPABASE_DB_URI,
    echo=False,
    pool_pre_ping=True,
    connect_args={
        "statement_cache_size": 0,
    },
)

AsyncSessionLocal: sessionmaker[AsyncSession] = sessionmaker(  # type: ignore
    engine,  # type: ignore
    class_=AsyncSession,
    expire_on_commit=False,
)  # type: ignore


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
