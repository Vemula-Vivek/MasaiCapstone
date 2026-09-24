"""FastAPI entry point for the offline-first Zepto support assistant."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag import AskResponse, ask, ingest_documents


class AskRequest(BaseModel):
    query: str = Field(min_length=1, description="Customer support question")


@asynccontextmanager
async def lifespan(_: FastAPI):
    ingest_documents()
    yield


app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    return ask(request.query)
