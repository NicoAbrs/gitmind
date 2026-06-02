# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gitmind is a RAG-powered chatbot that indexes a Git repository into Pinecone and lets users ask questions about the codebase via a Streamlit web UI. Embeddings use Gemini, generation uses Gemini chat.

## Setup

Install dependencies:
```
pip install -r requirements.txt
```

Required environment variables (create a `.env` file — it is gitignored):
- `GOOGLE_API_KEY` — from aistudio.google.com
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`

## Running locally

```
streamlit run app/ui.py
```

## Architecture

| File | Role |
|---|---|
| `app/pipeline.py` | All core logic: Pinecone setup, repo cloning, chunking, BM25 + Gemini embedding, upsert, query, answer generation |
| `app/ui.py` | Streamlit web UI — sidebar for indexing, chat interface for querying |

**Pipeline flow:** `pinecone_setup()` → `clone_repo()` → `process_files()` → `chunk_files()` → `embed_and_upsert()` → `query_index()` → `generate_answer()`

**Embeddings:** `gemini-embedding-001` (3072-dim, dense) + BM25 sparse vectors → hybrid dotproduct search in Pinecone

**Generation:** `gemini-2.5-flash` with full conversation history passed as `HumanMessage`/`AIMessage` list

**UI state:** `st.session_state.indexed` (bool), `st.session_state.index` (Pinecone index object), `st.session_state.messages` (chat history list)

## Deployment (Streamlit Cloud)

1. Push to GitHub
2. Go to share.streamlit.io → New app → select repo, set main file to `app/ui.py`
3. Add secrets: `GOOGLE_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
4. Deploy — share the generated URL
