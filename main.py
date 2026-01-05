from contextlib import asynccontextmanager
from enum import Enum

from fastapi import APIRouter, Depends, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from service.db_setup import get_db, init_db
from service.dependency import storage, vector_database
from service.file_service import File_Service

my_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application starting up...")
    my_resources["database_connection"] = "connected_to_database"
    await storage()
    await vector_database()
    await init_db()
    print("Database connection established.")
    yield
    print("Application shutting down...")
    if "database_connection" in my_resources:
        print("Closing database connection.")
        del my_resources["database_connection"]


app = FastAPI(lifespan=lifespan)
router = APIRouter()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChunkingMethod(str, Enum):
    SLIDING_WINDOW = "SLIDING_WINDOW"
    SEMANTIC_CHUNKING = "SEMANTIC_CHUNKING"


class SemanticMode(str, Enum):
    paragraph = "paragraph"
    section = "section"
    sentence = "sentence"


@router.post("/upload/file")
async def upload_file(
    file: UploadFile = File(...),
    chunking_method: ChunkingMethod = Form(ChunkingMethod.SLIDING_WINDOW),
    chunking_mode: SemanticMode = Form(SemanticMode.paragraph),
    store=Depends(storage),
    db=Depends(get_db),
    vdb=Depends(vector_database),
):
    return await File_Service.upload_single_file(
        file=file,
        store=store,
        db=db,
        vdb=vdb,
        chunking_method=chunking_method,
        chunking_mode=chunking_mode,
    )


@router.get("/get/all/docs")
async def get_all_docs(db=Depends(get_db)):
    return await File_Service.get_all_files(db)


class DocsCitations(BaseModel):
    query: str


@router.post("/get/docs-citations")
async def get_citations(
    data: DocsCitations,
    vdb=Depends(vector_database),
):
    return await File_Service.get_document_citations(query=data.query, vdb=vdb)


@router.post("/get/context-output")
async def get_context_output(
    data: DocsCitations,
    vdb=Depends(vector_database),
):
    return await File_Service.get_output_from_llm(query=data.query, vdb=vdb)


app.include_router(router)
