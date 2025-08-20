import click
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm

from .config import settings
from .indexer import NotesIndexer
from .search import NotesSearchEngine

console = Console()


@click.command()
@click.argument('command', nargs=1, required=False)
@click.argument('query', nargs=-1)
@click.option('--path', '-p', type=click.Path(exists=True, path_type=Path), 
              help='Path to notes directory to index')
@click.option('--force', '-f', is_flag=True, 
              help='Force reindex all files (ignore existing index)')
@click.option('--top-k', '-k', type=int, metavar='N',
              help='Number of search results to return (default: 5)')
@click.option('--verbose', '-v', is_flag=True, 
              help='Show full content of matched files')
@click.option('--files-only', '-l', is_flag=True, 
              help='Show only file paths and scores, no content')
@click.option('--json', '-j', is_flag=True, 
              help='Output search results in JSON format')
@click.option('--yes', '-y', is_flag=True, 
              help='Skip confirmation prompts')
@click.option('--version', is_flag=True, 
              help='Show version information')
def main(command, query, path, force, top_k, verbose, files_only, json, yes, version):
    """Personal Search Engine - Search through your notes and code snippets
    
    \b
    USAGE:
        psearch "your search query"        # Search notes
        psearch search "your query"        # Search notes (explicit)
        psearch index                      # Index notes directory
        psearch index --path ~/notes       # Index specific directory
        psearch interactive                # Interactive search mode
        psearch info                       # Show configuration
        psearch clear                      # Clear the index
        psearch --version                  # Show version
    
    \b
    EXAMPLES:
        psearch "python async functions"
        psearch "docker compose" -k 10
        psearch index --force
        psearch clear --yes
    
    \b
    OUTPUT OPTIONS:
        -v, --verbose     Show full content of matched files
        -l, --files-only  Show only file paths and scores
        -j, --json        Output results in JSON format
    """
    
    if version:
        console.print("psearch version 0.1.0")
        return
    
    # If no command given but query exists, treat as search
    if not command and not query:
        click.echo(click.get_current_context().get_help())
        return
    
    # If command looks like a search query (no recognized command)
    if command and command not in ['index', 'search', 'interactive', 'info', 'clear']:
        # Treat the command as part of the search query
        full_query = ' '.join([command] + list(query))
        do_search(full_query, top_k, verbose, files_only, json)
        return
    
    # Handle explicit commands
    if command == 'search':
        if not query:
            console.print("[red]Please provide a search query[/red]")
            return
        full_query = ' '.join(query)
        do_search(full_query, top_k, verbose, files_only, json)
    
    elif command == 'index':
        do_index(path, force)
    
    elif command == 'interactive':
        do_interactive()
    
    elif command == 'info':
        do_info()
    
    elif command == 'clear':
        do_clear(yes)
    
    else:
        # Default to search if query provided
        if query:
            full_query = ' '.join(query)
            do_search(full_query, top_k, verbose, files_only, json)


def do_search(query_str: str, top_k: Optional[int] = None, verbose: bool = False, files_only: bool = False, json_output: bool = False):
    """Execute search"""
    if not query_str.strip():
        console.print("[red]Please provide a search query[/red]")
        return
    
    search_engine = NotesSearchEngine()
    
    if not search_engine.vectorstore:
        console.print("[red]No index found. Please run 'psearch index' first.[/red]")
        return
    
    results = search_engine.search(query_str, top_k=top_k)
    search_engine.display_results(results, query_str, verbose=verbose, files_only=files_only, json_output=json_output)


def do_index(path: Optional[Path] = None, force: bool = False):
    """Index notes directory"""
    notes_dir = path or settings.notes_directory
    
    console.print(f"[bold]Personal Search Engine - Indexer[/bold]")
    console.print(f"Notes directory: {notes_dir}")
    console.print(f"Index directory: {settings.index_directory}")
    
    if not notes_dir.exists():
        console.print(f"[red]Error: Notes directory {notes_dir} does not exist[/red]")
        if Confirm.ask("Create directory?"):
            notes_dir.mkdir(parents=True)
            console.print(f"[green]Created {notes_dir}[/green]")
        else:
            return
    
    indexer = NotesIndexer(notes_dir=notes_dir)
    num_chunks = indexer.index(force_reindex=force)
    
    if num_chunks > 0:
        console.print(f"[bold green]✓ Indexing complete![/bold green]")


def do_interactive():
    """Interactive search mode"""
    console.print("[bold]Personal Search Engine - Interactive Mode[/bold]")
    console.print("Type 'quit' or 'exit' to leave\n")
    
    search_engine = NotesSearchEngine()
    
    if not search_engine.vectorstore:
        console.print("[red]No index found. Please run 'psearch index' first.[/red]")
        return
    
    while True:
        try:
            query = Prompt.ask("\n[bold blue]Search[/bold blue]")
            
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not query.strip():
                continue
            
            results = search_engine.search(query)
            search_engine.display_results(results, query)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def do_info():
    """Show configuration information"""
    console.print("[bold]Personal Search Engine - Configuration[/bold]\n")
    
    from rich.table import Table
    
    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Notes Directory", str(settings.notes_directory))
    table.add_row("Index Directory", str(settings.index_directory))
    table.add_row("Embedding Model", settings.embedding_model)
    table.add_row("Use Local Embeddings", str(settings.use_local_embeddings))
    table.add_row("Chunk Size", str(settings.chunk_size))
    table.add_row("Chunk Overlap", str(settings.chunk_overlap))
    table.add_row("Default Top K", str(settings.top_k))
    
    console.print(table)
    
    if settings.notes_directory.exists():
        file_count = sum(1 for _ in settings.notes_directory.rglob('*') if _.is_file())
        console.print(f"\n[dim]Files in notes directory: {file_count}[/dim]")
    
    if settings.index_directory.exists():
        console.print(f"[dim]Index exists: ✓[/dim]")
    else:
        console.print(f"[dim]Index exists: ✗[/dim]")


def do_clear(yes: bool = False):
    """Clear the search index"""
    if not yes:
        if not Confirm.ask(f"Clear index at {settings.index_directory}?"):
            console.print("[yellow]Cancelled[/yellow]")
            return
    
    if settings.index_directory.exists():
        import shutil
        shutil.rmtree(settings.index_directory)
        console.print(f"[green]✓ Index cleared[/green]")
    else:
        console.print("[yellow]No index to clear[/yellow]")


if __name__ == '__main__':
    main()