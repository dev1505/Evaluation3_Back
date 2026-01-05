import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from groq import AsyncGroq as groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = groq(api_key=GROQ_API_KEY)


class LlmService:
    @staticmethod
    async def get_structured_reranked_output(
        reranked_data: list,
        query: str,
    ):
        try:
            response = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": """
                            You are given a question and supporting document texts arranges after reranking, give detailed answer and answer should contain the information from the texts provided, also give some additional answer relates to the question asked that might  not be available in texts.
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"""
                            Question:
                            {query}

                            Context:
                            {json.dumps([{"text":data["text"],"score":data["score"], "final_score":data["final_score"]} for data in reranked_data], indent=2)}
                        """,
                    },
                ],
                temperature=0,
            )

            return {
                "data": (
                    response.choices[0].message.content
                    if response.choices[0].message.content
                    else ""
                ),
                "response_at": datetime.now(timezone.utc),
                "success": True,
            }

        except Exception:
            return {
                "data": "Error",
                "success": False,
            }
