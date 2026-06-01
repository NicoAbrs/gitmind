# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gitmind indexes Git repositories into a Pinecone vector database for semantic (hybrid dense + sparse) code search. The intended pipeline: clone repo → discover files → chunk with LangChain → embed with OpenAI → upsert into Pinecone.

## Setup

Install dependencies (note the typo in the filename):
```
pip install -r requiremens.txt
```

Required environment variables (create a `.env` file — it is gitignored):
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `OPENAI_API_KEY` (needed for `OpenAIEmbeddings`)

## Architecture

All logic lives in `app/pipeline.py`. The three implemented functions form the first half of the pipeline:

| Function | Role |
|---|---|
| `pinecone_setup()` | Createss (if absent) and returns a Pinecone serverless index — dense 1536-dim, dotproduct metric for hybrid search |
| `clone_repo(url, path)` | Wipes `path` and clones the given repo URL via GitPython |
| `process_files(path)` | Walks the clone, collecting `.py`, `.js`, `.ts`, `.md` files |

The remaining pipeline stages — chunking (`RecursiveCharacterTextSplitter` / `Language`), sparse encoding (`BM25Encoder`), embedding (`OpenAIEmbeddings`), and Pinecone upsert — are imported but not yet implemented. There is no `main()` or CLI entry point yet.