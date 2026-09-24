# Module 3: Zepto Support Assistant

## Run locally

```bash
python3 -m pip install -r requirements.txt
cd support_assistant
uvicorn main:app --host 127.0.0.1 --port 7860
```

`MOCK_LLM` is deliberately unset in these commands, so the required offline
mock path is used. `MOCK_LLM=1` has the same result. The first local run obtains
the open-source `all-MiniLM-L6-v2` model; it is then used locally for all
embeddings and no LLM provider, API key, or LLM network call is used.

Call the API from a second terminal:

```bash
curl -X POST http://127.0.0.1:7860/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the delivery fee below INR 149?"}'

{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01","doc_03","doc_05"],"confidence":1.0}% 
curl -X POST http://127.0.0.1:7860/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the capital of France?"}'  

{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}%   
```

## Mock-mode example transcript

Run the two commands above with `MOCK_LLM` unset and paste their raw JSON here
before submission. The policy answer begins with the deterministic required
prefix below; the exact order of the three `sources` follows local
cosine-similarity retrieval.

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders","sources":["doc_01","doc_03","doc_04"],"confidence":1.0}
```

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Architecture

```text
docs/doc_01.txt ... doc_08.txt
             |
             v
ingest_documents() -- SentenceTransformer(all-MiniLM-L6-v2)
             |                         |
             |                         v
             +------------------> ChromaDB: zepto_policy_chunks
                                           |
POST /ask -> classify_intent -> conditional route -> retrieve_and_answer
                  |                              |        |
                  +-> direct_answer              |        +-> AskResponse JSON
                                                 retrieval
```

1. **Ingestion:** `rag.py:ingest_documents()` reads the eight exact corpus
   files. Each document is one chunk and has an ID such as `doc_01`.
2. **Embedding:** the same function uses local `sentence-transformers` model
   `all-MiniLM-L6-v2` to produce normalized vector embeddings.
3. **Retrieval:** embeddings are persisted in ChromaDB collection
   `zepto_policy_chunks`. `retrieve_and_answer` calls `retrieve()` to embed a
   policy query and return the top three cosine-similarity chunks.
4. **Generation:** a LangGraph `StateGraph` first runs `classify_intent`, then
   routes to `retrieve_and_answer` or `direct_answer`. The FastAPI `/ask`
   endpoint returns the Pydantic `AskResponse` model: `answer`, `sources`, and
   `confidence`.

With `MOCK_LLM` unset or `1`, classification uses the required keyword
heuristic. Policy answers use the required `Based on the retrieved context:`
template and general questions use a fixed message. No LLM call occurs. With
the optional `MOCK_LLM=0` setting, those generation branches call the
Groq-compatible backend configured by `GROQ_API_KEY`; malformed JSON is retried
up to two additional times before a schema-marked error response is returned.
The structured prompt in `rag.py` explicitly contains the role, context, task,
format, length, a negative grounding constraint, and a few-shot example.

## Docker

Build and run the required local container baseline:

```bash
docker build -t zepto-support-assistant -f support_assistant/Dockerfile .
docker run --rm -p 7860:7860 zepto-support-assistant
```

The container defaults to `MOCK_LLM=1` and exposes `POST /ask` on port 7860.
