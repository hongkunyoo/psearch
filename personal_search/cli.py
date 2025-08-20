import click
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm

from .config import settings
from .indexer import NotesIndexer
from .search import NotesSearchEngine

console = Console()


@click.group()
@click.version_option(version='0.1.0')
def main():
    """Personal Search Engine - Search through your notes and code snippets"""
    pass


@main.command()
@click.option(
    '--path', '-p',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to notes directory'
)
@click.option(
    '--force', '-f',
    is_flag=True,
    help='Force reindex all files'
)
def index(path: Optional[Path], force: bool):
    """Index notes from specified directory"""
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


@main.command()
@click.argument('query', nargs=-1, required=True)
@click.option(
    '--top-k', '-k',
    type=int,
    default=None,
    help='Number of results to return'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show full content of results'
)
def search(query: tuple, top_k: Optional[int], verbose: bool):
    """Search through indexed notes"""
    query_str = ' '.join(query)
    
    if not query_str.strip():
        console.print("[red]Please provide a search query[/red]")
        return
    
    search_engine = NotesSearchEngine()
    
    if not search_engine.vectorstore:
        console.print("[red]No index found. Please run 'psearch index' first.[/red]")
        return
    
    results = search_engine.search(query_str, top_k=top_k)
    search_engine.display_results(results, query_str, verbose=verbose)


@main.command()
def interactive():
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


@main.command()
def info():
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


@main.command()
@click.option(
    '--yes', '-y',
    is_flag=True,
    help='Skip confirmation'
)
def clear(yes: bool):
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