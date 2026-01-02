from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from service.dependency import database, storage, vector_database
from service.file_service import File_Service

# from service.llm_service import

my_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application starting up...")
    my_resources["database_connection"] = "connected_to_database"
    database()
    storage()
    await vector_database()
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
    db=Depends(database),
    vdb=Depends(vector_database),
):
    return await File_Service.upload_single_file(file=file, store=store, db=db, vdb=vdb)


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
# def main():
#     print("Hello from eval3-back!")


# if __name__ == "__main__":
#     main()
