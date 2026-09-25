# Zepto Data & AI Platform

This repository contains one connected AI/ML engineering project with three
capabilities: a raw-to-relational data pipeline, an end-to-end analytics and
modeling workflow, and a grounded GenAI policy support API.

## Repository structure

```text
data_pipeline/      Books to Scrape ingestion, cleaning, SQLite, and SQL analysis
analytics/          Titanic EDA, classification, regression, and saved model
support_assistant/  Zepto policy RAG assistant with FastAPI and Docker
requirements.txt    Consolidated dependencies for all three modules
```

## Setup

This project uses one consolidated `requirements.txt` at the repository root.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/MasaiCapstone.git
cd MasaiCapstone
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run the modules

### 1. Data pipeline

```bash
python3 data_pipeline/data_pipeline.py
```

The script scrapes all pages in the Classics, Philosophy, and Fiction
categories from Books to Scrape. It cleans catalog fields, converts prices with
the fixed assignment rate of 1 GBP = 105.50 INR, recreates the normalized
SQLite snapshot, saves five SQL queries and their results, and verifies that
the SQL JOIN matches the equivalent pandas merge.

See [data_pipeline/README.md](data_pipeline/README.md) for the schema and
cleaning decisions.

### 2. Analytics pipeline

```bash
python3 analytics/pipeline.py
```

This is the repository's only Titanic network/cache load through
`sns.load_dataset('titanic')`. It immediately saves `analytics/titanic.csv` as
the committed offline fallback, then performs cleaning, EDA, charts,
classification, imbalance comparison, Random Forest tuning, regression, and
saved-pipeline reload validation.

See [analytics/README.md](analytics/README.md) for missing-value decisions,
chart interpretations, model metrics, and the final recommendation.

### 3. Support assistant

```bash
unset MOCK_LLM
cd support_assistant
uvicorn main:app --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860/docs` for the interactive API, or call the API:

```bash
curl -X POST http://127.0.0.1:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the delivery fee below INR 149?"}'
```

With `MOCK_LLM` unset or set to `1`, the required deterministic mock mode is
used: ChromaDB retrieval still runs locally, but no request is sent to any LLM
provider and no API key is needed. `MOCK_LLM=0` enables the optional
Groq-compatible real-LLM path and requires `GROQ_API_KEY`.

See [support_assistant/README.md](support_assistant/README.md) for the RAG
architecture, mock-mode API transcripts, Docker commands, and optional real
LLM behavior.

## Design summary

- **Data pipeline:** Uses runtime category discovery and page traversal to
  collect catalog data. A two-table SQLite schema separates categories from
  books through a primary/foreign key relationship.
- **Analytics pipeline:** Keeps EDA and modeling connected to one Titanic
  source. Modeling preprocessing is encapsulated in train-only scikit-learn
  pipelines to avoid test-data leakage.
- **Support assistant:** Uses eight local Zepto policy documents, local
  `all-MiniLM-L6-v2` embeddings, ChromaDB similarity retrieval, LangGraph
  intent routing, Pydantic response validation, and FastAPI delivery.

## Generated artifacts and secrets

The repository includes required generated outputs such as `analytics/titanic.csv`,
analytics charts, the saved `.joblib` pipeline, and SQL query results. Local
runtime data such as `.venv/`, `__pycache__/`, SQLite databases, and the
ChromaDB persistence directory are ignored. Do not commit API keys or other
secrets.
