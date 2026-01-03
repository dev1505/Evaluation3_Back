from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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


origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@router.post("/upload/file")
async def upload_file(
    file: UploadFile = File(...),
    store=Depends(storage),
    db=Depends(get_db),
    chunking_method=str,
    vdb=Depends(vector_database),
):
    return await File_Service.upload_single_file(
        file=file,
        store=store,
        db=db,
        vdb=vdb,
        chunking_method=chunking_method if chunking_method else "SLIDING_WINDOW",  # type: ignore
    )


@router.post("/api/verify-citation")
async def get_citations(
    query: str,
    vdb=Depends(vector_database),
):
    return await File_Service.get_document_citations(query=query, vdb=vdb)


@router.post("/get/context-output")
async def get_context_output(
    query: str,
    vdb=Depends(vector_database),
):
    return await File_Service.get_output_from_llm(query=query, vdb=vdb)


app.include_router(router)
