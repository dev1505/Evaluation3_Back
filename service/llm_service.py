import json
import os
from typing import List

from dotenv import load_dotenv
from google import genai
from groq import AsyncGroq as groq
from pydantic import BaseModel

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# client = genai.Client(api_key=GROQ_API_KEY)
client = groq(api_key=GROQ_API_KEY)


class DocumentChunk(BaseModel):
    score: float
    text: str
    document_id: str
    file_name: str
    page_start: int
    page_end: int
    section_path: List[str]
    chunk_index: int
    uploaded_at: str
    final_score: float


class RerankedResponse(BaseModel):
    results: List[DocumentChunk]


class LlmService:
    @staticmethod
    async def get_structured_reranked_output(
        reranked_data: list,
        query: str,
    ):
        response = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant.\n"
                        "You MUST return valid JSON only.\n"
                        "Do not include explanations outside JSON.\n"
                        "Use the provided context.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
                    Answer the question using the context below.

                    Return JSON in this exact format:
                    {{
                    "answer": string,
                    "score": float
                    "text": str
                    "document_id": str
                    "file_name": str
                    "page_start": int
                    "page_end": int
                    "section_path": List[str]
                    "chunk_index": int
                    "uploaded_at": str
                    "final_score": float
                    "citations": [
                        {{
                        "chunk_index": number,
                        "reason": string
                        }}
                    ]
                    }}

                    Question:
                    {query}

                    Context:
                    {json.dumps(reranked_data, indent=2)}
                    """,
                },
            ],
            temperature=0,
        )

        raw_text = response.choices[0].message.content
        parsed = json.loads(raw_text)
        return parsed
