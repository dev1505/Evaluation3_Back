import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastembed import TextEmbedding

from service.parsers import Parsers

SemanticMode = Literal["paragraph", "sentence", "section"]


@dataclass
class ChunkMetadata:
    document_id: str
    page_start: int
    page_end: int
    section_path: List[str]
    chunk_index: int
    uploaded_at: datetime


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata


HEADER_REGEX = re.compile(r"^(#{1,6}\s+|[A-Z][A-Za-z0-9\s]{2,50}:$)")
LIST_ITEM_REGEX = re.compile(r"^\s*(\d+\.|\-|\*)\s+")
TABLE_ROW_REGEX = re.compile(r"\|.*\|")


_embedding_model = TextEmbedding()


def embed_chunk(chunk: Chunk):
    vector = list(_embedding_model.embed(chunk.text))[0]
    return {
        "chunk": chunk,
        "embedding": vector,
    }


def embed_query(query: str):
    return list(_embedding_model.embed(query))[0]


def structural_units(pages: List[str]) -> List[Tuple[str, int, Optional[str]]]:
    units: List[Tuple[str, int, Optional[str]]] = []
    current_section: Optional[str] = None

    for page_num, page_text in enumerate(pages, start=1):
        lines = [l.rstrip() for l in page_text.splitlines()]
        paragraph_lines: List[str] = []

        for line in lines:
            if not line.strip():
                if paragraph_lines:
                    units.append(
                        (" ".join(paragraph_lines).strip(), page_num, current_section)
                    )
                    paragraph_lines = []
                continue

            if HEADER_REGEX.match(line):
                if paragraph_lines:
                    units.append(
                        (" ".join(paragraph_lines).strip(), page_num, current_section)
                    )
                    paragraph_lines = []

                current_section = line.strip()
                units.append((current_section, page_num, current_section))
                continue

            if LIST_ITEM_REGEX.match(line):
                paragraph_lines.append(line.strip())
                continue

            if TABLE_ROW_REGEX.match(line):
                paragraph_lines.append(line.strip())
                continue

            # Normal wrapped text line → merge safely
            paragraph_lines.append(line.strip())

        if paragraph_lines:
            units.append((" ".join(paragraph_lines).strip(), page_num, current_section))

    return units


def semantic_chunker(
    document_id: str,
    max_chunk_size: int,
    file_bytes: bytes,
    mode: SemanticMode = "paragraph",
) -> List[Dict[str, Any]]:

    pages = Parsers.pdf_parser_from_upload(file_bytes=file_bytes)
    if not pages:
        return []

    units = structural_units(pages)

    embedded_chunks: List[Dict[str, Any]] = []
    current_text = ""
    current_pages: List[int] = []
    current_section_path: List[str] = []
    chunk_index = 0

    def flush_chunk():
        nonlocal current_text, current_pages, chunk_index
        if not current_text.strip():
            return

        chunk = Chunk(
            text=current_text.strip(),
            metadata=ChunkMetadata(
                document_id=document_id,
                page_start=min(current_pages),
                page_end=max(current_pages),
                section_path=current_section_path.copy(),
                chunk_index=chunk_index,
                uploaded_at=datetime.now(timezone.utc),
            ),
        )
        embedded_chunks.append(embed_chunk(chunk))
        chunk_index += 1
        current_text = ""
        current_pages = []

    for text, page, section in units:
        if section:
            current_section_path = [section]

        if mode == "section":
            parts = [text]

        elif mode == "sentence":
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        else:
            parts = [text]

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if current_text and len(current_text) + len(part) > max_chunk_size:
                flush_chunk()

            current_text += ("\n\n" if current_text else "") + part
            current_pages.append(page)

    flush_chunk()
    return embedded_chunks


def sliding_window_chunker(
    document_id: str,
    chunk_size: int,
    file_bytes: bytes,
    overlap: int,
) -> List[Dict[str, Any]]:

    pages = Parsers.pdf_parser_from_upload(file_bytes=file_bytes)

    if not pages:
        return []

    units = structural_units(pages)
    flat_units = [(text, page) for text, page, _ in units]
    flat_text = "\n".join(text for text, _ in flat_units)

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    embedded_chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(flat_text):
        end = start + chunk_size
        chunk_text = flat_text[start:end]

        pages_in_chunk = [page for text, page in flat_units if text in chunk_text]

        chunk = Chunk(
            text=chunk_text,
            metadata=ChunkMetadata(
                document_id=document_id,
                page_start=min(pages_in_chunk) if pages_in_chunk else 1,
                page_end=max(pages_in_chunk) if pages_in_chunk else 1,
                section_path=[],
                chunk_index=chunk_index,
                uploaded_at=datetime.now(timezone.utc),
            ),
        )

        embedded_chunks.append(embed_chunk(chunk))

        chunk_index += 1
        start += step

    return embedded_chunks
