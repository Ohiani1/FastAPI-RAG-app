"""
FastAPI service.

    uvicorn api.main:app --reload
    curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
         -d '{"question": "How do I make a query parameter optional?"}'
"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.rag import answer_question, to_dict

app = FastAPI(title="Ask the FastAPI Docs")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = []   # earlier turns, for follow-up questions


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    # Plain `def`: the DB and LLM clients are synchronous, so FastAPI runs this in a
    # thread pool instead of blocking the event loop. Streaming comes with the frontend.
    result = answer_question(req.question, [m.model_dump() for m in req.history])
    return to_dict(result)
