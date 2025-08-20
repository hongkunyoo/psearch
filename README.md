# Personal Search Engine

A CLI-based semantic search engine for your personal notes and code snippets.

## Features

- Semantic search using natural language queries
- Indexes text files, markdown, code files, and more
- Local embeddings (no API keys required)
- Fast vector similarity search with ChromaDB
- Rich terminal output with syntax highlighting
- Interactive search mode

## Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### 1. Index your notes

```bash
# Index default notes directory (~/notes)
uv run python -m personal_search index

# Index a specific directory
uv run python -m personal_search index --path /path/to/your/notes

# Force reindex all files
uv run python -m personal_search index --force
```

### 2. Search your notes

```bash
# Search with a query
uv run python -m personal_search search "python async functions"

# Get more results
uv run python -m personal_search search "docker compose" --top-k 10

# Show full content
uv run python -m personal_search search "kubernetes deployment" --verbose
```

### 3. Interactive mode

```bash
# Start interactive search
uv run python -m personal_search interactive
```

### Other commands

```bash
# Show configuration
uv run python -m personal_search info

# Clear the index
uv run python -m personal_search clear
```

## Configuration

Copy `.env.example` to `.env` and customize settings:

- `NOTES_DIRECTORY`: Directory containing your notes
- `INDEX_DIRECTORY`: Where to store the search index
- `EMBEDDING_MODEL`: Which embedding model to use
- `TOP_K`: Default number of search results

## Supported File Types

- Text files (`.txt`)
- Markdown (`.md`)
- Python (`.py`)
- JavaScript (`.js`)
- JSON (`.json`)
- YAML (`.yaml`, `.yml`)
- Shell scripts (`.sh`)
- SQL (`.sql`)
- And more...