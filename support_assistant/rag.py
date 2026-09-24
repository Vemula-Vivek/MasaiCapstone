"""Offline-first LangGraph RAG workflow for the Zepto policy corpus."""

import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
import requests
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, START, StateGraph


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policy_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
POLICY_KEYWORDS = (
    "delivery", "return", "refund", "membership", "tracking", "cancel",
    "gift card", "support hours" )

# Role -> Context -> Task -> Format -> Length, with a negative constraint and
# a few-shot example. It is used only when MOCK_LLM=0.
STRUCTURED_PROMPT_TEMPLATE = """Role:
You are Zepto's policy support assistant.

Context:
{context}

Task:
Answer the customer's query: {query}
Do not answer using information not present in the provided context.

Format:
Return only JSON with exactly these fields:
{{"answer": "string", "sources": ["document IDs"], "confidence": 0.0}}

Length:
Keep the answer under 90 words.

Few-shot example:
Context: doc_01 says orders below INR 149 have a flat INR 25 delivery fee.
Query: What is the delivery fee below INR 149?
JSON: {{"answer":"Orders below INR 149 have a flat INR 25 delivery fee.","sources":["doc_01"],"confidence":0.95}}
"""


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_chunks: list[dict[str, str]]
    answer: str
    sources: list[str]
    confidence: float


_model: SentenceTransformer | None = None
_collection = None


def mock_mode() -> bool:
    """Mock mode is the safe default unless explicitly disabled."""
    return os.getenv("MOCK_LLM", "1") != "0"


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def ingest_documents() -> None:
    """Embed the eight policy documents and upsert one chunk per document."""
    documents = []
    ids = []
    metadatas = []

    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        document_id = path.stem
        ids.append(document_id)
        documents.append(path.read_text(encoding="utf-8").strip())
        metadatas.append({"document_id": document_id, "filename": path.name})

    if len(documents) != 8:
        raise RuntimeError(f"Expected 8 corpus documents, found {len(documents)}")

    embeddings = get_embedding_model().encode(
        documents,
        normalize_embeddings=True,
    ).tolist()
    get_collection().upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def retrieve(query: str, top_k: int = 3) -> list[dict[str, str]]:
    """Embed the query locally and retrieve the top ChromaDB cosine matches."""
    query_embedding = get_embedding_model().encode(
        [query], normalize_embeddings=True
    ).tolist()
    results = get_collection().query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "id": metadata["document_id"],
            "text": document,
            "distance": str(distance),
        }
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        )
    ]


def call_real_llm(prompt: str) -> str:
    """Optional Groq-compatible real LLM call, used only with MOCK_LLM=0."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("MOCK_LLM=0 requires GROQ_API_KEY")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def validated_real_answer(prompt: str, fallback_sources: list[str]) -> AskResponse:
    """Retry malformed optional real-LLM JSON up to two additional times."""
    retry_instruction = "\nYour previous response was invalid. Return valid JSON only."
    last_error = "unknown validation error"

    for _ in range(3):
        raw_output = call_real_llm(prompt + retry_instruction)
        try:
            return AskResponse.model_validate(json.loads(raw_output))
        except (json.JSONDecodeError, ValidationError) as error:
            last_error = str(error)

    return AskResponse(
        answer=f"ERROR: Real LLM output failed schema validation: {last_error}",
        sources=fallback_sources,
        confidence=0.0,
    )


def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]
    if mock_mode():
        intent = (
            "policy_question"
            if any(keyword in query.lower() for keyword in POLICY_KEYWORDS)
            else "general_question"
        )
    else:
        prompt = (
            "Classify this query as policy_question or general_question. "
            f"Query: {query}. Return JSON only with key intent."
        )
        try:
            intent = json.loads(call_real_llm(prompt))["intent"]
            if intent not in {"policy_question", "general_question"}:
                raise ValueError("unexpected intent")
        except (json.JSONDecodeError, KeyError, ValueError):
            intent = "general_question"
    return {"intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    chunks = retrieve(state["query"])
    sources = [chunk["id"] for chunk in chunks]

    if mock_mode():
        answer = f"Based on the retrieved context: {chunks[0]['text'][:200]}"
        return {
            "retrieved_chunks": chunks,
            "answer": answer,
            "sources": sources,
            "confidence": 1.0,
        }

    context = "\n\n".join(f"[{chunk['id']}] {chunk['text']}" for chunk in chunks)
    response = validated_real_answer(
        STRUCTURED_PROMPT_TEMPLATE.format(context=context, query=state["query"]),
        sources,
    )
    return {
        "retrieved_chunks": chunks,
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
    }


def direct_answer(state: GraphState) -> GraphState:
    if mock_mode():
        return {
            "answer": "I can only answer questions about Zepto policies right now.",
            "sources": [],
            "confidence": 1.0,
        }

    response = validated_real_answer(
        STRUCTURED_PROMPT_TEMPLATE.format(context="No policy context was retrieved.", query=state["query"]),
        [],
    )
    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
    }


def route_after_classification(state: GraphState) -> str:
    return state["intent"]


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


def ask(query: str) -> AskResponse:
    result = build_graph().invoke({"query": query})
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )
