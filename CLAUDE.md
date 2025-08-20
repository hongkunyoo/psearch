# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal search engine for searching through developer notes and code snippets stored as text files. The system indexes notes from a directory (e.g., `~/notes/`) and provides semantic search capabilities via CLI.

## Architecture

### Technology Stack
- **Python** with LangChain/LangGraph for semantic search
- **Vector Database**: ChromaDB or FAISS for embedding storage
- **Embeddings**: OpenAI or local models (e.g., Sentence Transformers)
- **CLI**: Click or Typer for command-line interface

### Core Components
1. **Indexer**: Scans notes directory and creates/updates vector embeddings
2. **Search Engine**: Performs semantic search using natural language queries
3. **CLI Interface**: Simple terminal commands for indexing and searching

## Development Commands

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with uv
uv venv

# Install dependencies
uv pip install -r requirements.txt

# Run the indexer
uv run python -m personal_search index --path ~/notes/

# Search notes
uv run python -m personal_search search "your query here"

# Run tests
uv run pytest tests/

# Lint code
uv run ruff check .
uv run black --check .
```

## Project Requirements

- Index text files from a specified directory (default: `~/notes/`)
- Support natural language queries for semantic search
- CLI-only interface (no web UI needed)
- Return relevant notes ranked by similarity
- Handle code snippets and technical documentation effectively
- Support incremental indexing for new/modified files
