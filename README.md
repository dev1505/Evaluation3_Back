# eval3-back

## Project Description
`eval3-back` is a robust backend service built with FastAPI, designed to facilitate Retrieval Augmented Generation (RAG) applications. It provides functionalities for seamless document processing, intelligent chunking, vector embedding generation, and efficient interaction with vector databases and Large Language Models (LLMs). This service acts as a core component for systems requiring contextual understanding and generation from various document types.

## Key Features

*   **File Upload & Processing**: Supports uploading various document types (e.g., PDF, DOCX) for ingestion.
*   **Intelligent Document Chunking**: Automatically breaks down documents into manageable chunks using configurable methods like "SLIDING_WINDOW" to optimize for embedding and retrieval.
*   **Vector Embeddings Generation**: Generates high-quality vector embeddings for document chunks, enabling semantic search and retrieval.
*   **Vector Database Integration**: Utilizes Qdrant as a vector store for efficient storage and retrieval of document embeddings.
*   **Citation Verification**: Provides an API to verify information against the ingested documents, ensuring factual accuracy.
*   **Contextual AI Responses**: Integrates with leading LLMs (Google GenAI, Groq) to provide intelligent, context-aware responses based on retrieved document chunks.
*   **Database Management**: Leverages SQLAlchemy and Alembic for robust relational database management, supporting Supabase integration.

## API Endpoints

The service exposes the following API endpoints:

### 1. Upload File
- **Endpoint**: `POST /upload/file`
- **Description**: Uploads a single file, processes it (chunks, embeds), and stores its information and embeddings in the vector database.
- **Parameters**:
    - `file`: The document file to upload (`UploadFile`).
    - `chunking_method`: Optional. Specifies the chunking strategy (e.g., "SLIDING_WINDOW"). Defaults to "SLIDING_WINDOW".

### 2. Verify Citation
- **Endpoint**: `POST /api/verify-citation`
- **Description**: Verifies a given query against the documents stored in the vector database to find relevant citations.
- **Parameters**:
    - `query`: The text query to verify (`str`).

### 3. Get Contextual Output
- **Endpoint**: `POST /get/context-output`
- **Description**: Retrieves a contextual response from an LLM based on the provided query and information retrieved from the vector database.
- **Parameters**:
    - `query`: The text query for which to get a contextual response (`str`).

## Technologies Used

*   **FastAPI**: A modern, fast (high-performance) web framework for building APIs with Python 3.7+.
*   **uv**: A fast Python package installer and dependency resolver.
*   **Google GenAI**: For interacting with Google's generative AI models.
*   **Groq**: For fast language model inference.
*   **Qdrant**: High-performance, scalable vector database.
*   **SQLAlchemy**: Python SQL toolkit and Object Relational Mapper (ORM).
*   **Alembic**: Lightweight database migration tool for SQLAlchemy.
*   **asyncpg**: A fast PostgreSQL client library for Python.
*   **pdf2image**: Converts PDF pages into PIL Image objects.
*   **Pillow**: The friendly PIL fork (Python Imaging Library).
*   **PyPDF**: A pure-Python PDF library.
*   **pytesseract**: Python wrapper for Google's Tesseract-OCR Engine.
*   **python-docx**: Library for creating and updating Microsoft Word files (.docx).
*   **Supabase**: For database hosting and other backend services.

## Setup and Installation

### Prerequisites

*   Python 3.12 or newer
*   `uv` package manager (install via `pip install uv`)

### Installation Steps

1.  **Clone the repository**:
    ```bash
    git clone [repository-url]
    cd eval3-back
    ```
2.  **Install dependencies**:
    ```bash
    uv sync
    ```

### Running the Application

1.  **Start the FastAPI application**:
    ```bash
    uvicorn main:app --reload
    ```
    The application will be accessible at `http://127.0.0.1:8000`.

### Database Setup

This project uses Alembic for database migrations. You will need to configure your database connection (e.g., in an environment variable or configuration file) before running migrations.

1.  **Initialize/Upgrade Database**:
    ```bash
    alembic upgrade head
    ```

**Note**: Ensure your environment variables for database connections, LLM API keys, and Qdrant configurations are set up correctly before running the application.
