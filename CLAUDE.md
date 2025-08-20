# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal search engine for searching through developer notes and code snippets stored as text files. The system indexes notes from a directory (e.g., `~/notes/`) and provides semantic search capabilities via CLI.

## Architecture

### Technology Stack
- **Python 3.9+** with LangChain for document processing
- **Vector Database**: ChromaDB for persistent embedding storage
- **Embeddings**: HuggingFace Sentence Transformers (local, no API needed)
- **CLI**: Click for command-line interface
- **UI**: Rich for beautiful terminal output with syntax highlighting

### Core Components
1. **`indexer.py`**: NotesIndexer class that scans notes directory and creates/updates vector embeddings
2. **`search.py`**: NotesSearchEngine class that performs semantic search using natural language queries
3. **`cli.py`**: Click-based CLI with commands for indexing, searching, and configuration
4. **`config.py`**: Pydantic settings management with .env support

### File Structure
```
personal_search/
├── __init__.py
├── __main__.py      # Entry point for python -m personal_search
├── cli.py           # CLI commands (index, search, interactive, info, clear)
├── config.py        # Settings and configuration
├── indexer.py       # Document loading and vector indexing
└── search.py        # Search engine and result display
```

## Development Commands

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with uv
uv venv

# Install package in editable mode
uv pip install -e .

# Usage as binary command (after installation)
psearch "python async functions"       # Direct search
psearch index                         # Index notes
psearch index --path ~/custom/notes   # Index custom directory
psearch interactive                    # Interactive mode
psearch info                          # Show configuration
psearch clear --yes                   # Clear index

# Or run directly without installation
uv run python -m personal_search "query"
uv run python -m personal_search index

# Run tests (when implemented)
uv run pytest tests/

# Lint code
uv run ruff check .
uv run black --check .
```

## Configuration

Configuration via `.env` file or environment variables:
- `NOTES_DIRECTORY`: Directory containing notes (default: `~/notes`)
- `INDEX_DIRECTORY`: Vector database storage (default: `~/.personal_search/index`)
- `EMBEDDING_MODEL`: HuggingFace model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `USE_LOCAL_EMBEDDINGS`: Use local embeddings (default: `true`)
- `CHUNK_SIZE`: Text chunk size for indexing (default: `1000`)
- `CHUNK_OVERLAP`: Overlap between chunks (default: `200`)
- `TOP_K`: Default number of search results (default: `5`)

## Supported File Types

The indexer automatically scans for:
- `.txt` - Plain text files
- `.md` - Markdown files
- `.py` - Python scripts
- `.js` - JavaScript files
- `.json` - JSON configuration
- `.yaml`, `.yml` - YAML files
- `.sh` - Shell scripts
- `.sql` - SQL queries
- Extension-less files that contain readable text content

## Key Implementation Details

- Uses RecursiveCharacterTextSplitter for document chunking
- Embeddings are generated locally using CPU (no GPU required)
- ChromaDB persists index to disk for fast restarts
- Rich terminal output with syntax highlighting for code snippets
- Similarity scores shown for search results (lower is better)
