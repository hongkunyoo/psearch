# Personal Search Engine

A fast, local CLI tool for semantic search through your personal notes and code snippets.

## Features

- 🔍 **Natural Language Search**: Search using queries like "python async functions" 
- 🚀 **Fast & Local**: Uses local embeddings - no API keys required
- 💾 **Persistent Index**: ChromaDB vector database stores embeddings on disk
- 🎨 **Rich Terminal UI**: Syntax highlighting for code snippets
- 📝 **Multiple Formats**: Supports `.txt`, `.md`, `.py`, `.js`, `.json`, `.yaml`, and more

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd personal-search-engine

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install psearch in editable mode
uv pip install -e .
```

### Basic Usage

```bash
# First time: Index your notes
psearch index

# Search your notes
psearch "python async await"
psearch "docker compose"
psearch "git rebase"

# More results
psearch "kubernetes" -k 10

# Show full content
psearch "terraform" --verbose
```

## Commands

### Search (default)
```bash
psearch "your search query"        # Direct search
psearch search "your query"        # Explicit search command
psearch "query" -k 10              # Get top 10 results
psearch "query" --verbose          # Show full content
```

### Index
```bash
psearch index                      # Index default notes directory
psearch index --path ~/my-notes   # Index custom directory
psearch index --force             # Force reindex all files
```

### Interactive Mode
```bash
psearch interactive               # Enter interactive search mode
```

### Other Commands
```bash
psearch info                      # Show configuration
psearch clear                     # Clear the index
psearch clear --yes              # Clear without confirmation
psearch --version                # Show version
psearch --help                   # Show help
```

## Configuration

Create a `.env` file to customize settings:

```bash
# Copy example config
cp .env.example .env
```

Available settings:
- `NOTES_DIRECTORY`: Where your notes are stored (default: `~/notes`)
- `INDEX_DIRECTORY`: Where to store the search index (default: `~/.personal_search/index`)
- `EMBEDDING_MODEL`: Which model to use for embeddings
- `TOP_K`: Default number of results (default: 5)
- `CHUNK_SIZE`: Size of text chunks for indexing (default: 1000)

## Examples

### Quick Search Workflow
```bash
# Index your notes (one time)
psearch index

# Search for async Python code
psearch "async await python"

# Search for Docker commands
psearch "docker build push"

# Search for git workflows
psearch "git cherry pick"
```

### Interactive Mode
```bash
psearch interactive
# Then type queries interactively
# Type 'quit' to exit
```

## Supported File Types

- `.txt` - Plain text files
- `.md` - Markdown documentation
- `.py` - Python scripts
- `.js` - JavaScript files
- `.json` - JSON configuration
- `.yaml`, `.yml` - YAML files
- `.sh` - Shell scripts
- `.sql` - SQL queries

## How It Works

1. **Indexing**: Scans your notes directory and creates vector embeddings using Sentence Transformers
2. **Storage**: Stores embeddings in ChromaDB (persisted at `~/.personal_search/index`)
3. **Search**: Converts your query to embeddings and finds similar documents using cosine similarity
4. **Display**: Shows results with syntax highlighting and relevance scores

## Development

```bash
# Install dev dependencies
uv pip install -r requirements.txt

# Run directly without installation
uv run python -m personal_search "query"

# Format code
uv run black personal_search/
uv run ruff check personal_search/
```

## License

MIT
