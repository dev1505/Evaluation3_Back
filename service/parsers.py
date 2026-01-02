import io
from typing import List, Union

import pytesseract
from docx import Document
from pdf2image import convert_from_bytes
from PIL import Image
from pypdf import PdfReader


class Parsers:

    @staticmethod
    def parse_uploaded_docs(mime_type, file_bytes):
        if mime_type == "application/pdf":
            return {
                "data": Parsers.pdf_parser_from_upload(file_bytes == file_bytes),
                "success": True,
            }
        else:
            return {
                "data": "File of this type is not supported",
                "success": False,
            }

    @staticmethod
    def pdf_parser_from_upload(file_bytes: bytes, ocr_threshold: int = 50) -> List[str]:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text: List[str] = []
        images = convert_from_bytes(file_bytes)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if len(text.strip()) < ocr_threshold:
                image = images[i]
                ocr_text = pytesseract.image_to_string(image)
                text = ocr_text
            pages_text.append(text.strip())
        return pages_text

    # @staticmethod
    # async def image_parser_from_upload(image_bytes) -> str:
    #     try:
    #         image = Image.open(io.BytesIO(image_bytes))
    #         text = pytesseract.image_to_string(image)
    #         return text.strip()
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to parse image: {e}")

    @staticmethod
    async def word_parser_from_upload(file_bytes) -> str:
        try:
            file_bytes = file_bytes
            doc = Document(io.BytesIO(file_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to parse Word document: {e}")
