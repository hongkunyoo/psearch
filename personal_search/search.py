from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from langchain_chroma import Chroma
from langchain.schema import Document
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from .config import settings
from .indexer import NotesIndexer

console = Console()


class SearchResult:
    def __init__(self, document: Document, score: float):
        self.document = document
        self.score = score
        self.content = document.page_content
        self.metadata = document.metadata
        self.source = Path(document.metadata.get('source', 'unknown'))
        self.filename = document.metadata.get('filename', 'unknown')
        self.modified = document.metadata.get('modified', '')
    
    def __repr__(self):
        return f"SearchResult(file={self.filename}, score={self.score:.3f})"


class NotesSearchEngine:
    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = index_dir or settings.index_directory
        self.indexer = NotesIndexer(index_dir=self.index_dir)
        self.vectorstore = self.indexer.get_vectorstore()
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if not self.vectorstore:
            console.print("[red]No index found. Please run 'psearch index' first.[/red]")
            return []
        
        top_k = top_k or settings.top_k
        
        try:
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=top_k,
                filter=filter_dict
            )
            
            search_results = [
                SearchResult(doc, score)
                for doc, score in results_with_scores
            ]
            
            search_results.sort(key=lambda x: x.score)
            
            return search_results
            
        except Exception as e:
            console.print(f"[red]Search error: {e}[/red]")
            return []
    
    def display_results(self, results: List[SearchResult], query: str, verbose: bool = False, files_only: bool = False, json_output: bool = False):
        if not results:
            if json_output:
                print(json.dumps({"query": query, "results": []}))
            else:
                console.print("[yellow]No results found.[/yellow]")
            return
        
        if json_output:
            self._display_json_results(results, query, verbose)
        elif files_only:
            self._display_files_only(results, query)
        else:
            console.print(f"\n[bold blue]Found {len(results)} results for: '{query}'[/bold blue]\n")
            
            for i, result in enumerate(results, 1):
                self._display_single_result(result, i, verbose)
    
    def _display_files_only(self, results: List[SearchResult], query: str):
        """Display only file paths and scores, not content"""
        console.print(f"\n[bold blue]Found {len(results)} results for: '{query}'[/bold blue]\n")
        
        for i, result in enumerate(results, 1):
            score_str = f"[dim](Score: {result.score:.3f})[/dim]"
            console.print(f"[bold cyan]{i:2}.[/bold cyan] {result.source} {score_str}")
    
    def _display_json_results(self, results: List[SearchResult], query: str, verbose: bool = False):
        """Display results in JSON format"""
        json_results = []
        
        for result in results:
            result_data = {
                "filename": result.filename,
                "path": str(result.source),
                "score": round(result.score, 3),
                "modified": result.modified,
                "content": result.content if verbose else (result.content[:500] + "..." if len(result.content) > 500 else result.content)
            }
            json_results.append(result_data)
        
        output = {
            "query": query,
            "total_results": len(results),
            "results": json_results
        }
        
        print(json.dumps(output, indent=2, ensure_ascii=False))
    
    def _display_single_result(self, result: SearchResult, index: int, verbose: bool = False):
        # Escape brackets in filename to prevent Rich markup parsing issues
        escaped_filename = result.filename.replace('[', r'\[').replace(']', r'\]')
        title = f"[{index}] {escaped_filename} (Score: {result.score:.3f})"
        
        content_preview = result.content[:500] + "..." if len(result.content) > 500 else result.content
        
        if verbose:
            content_to_show = result.content
        else:
            content_to_show = content_preview
        
        file_ext = result.source.suffix.lower()
        lexer = self._get_lexer_for_extension(file_ext)
        
        if lexer:
            syntax = Syntax(content_to_show, lexer, theme="monokai", line_numbers=True)
            panel_content = syntax
        else:
            # Escape Rich markup in plain text content to prevent parsing errors
            from rich.text import Text
            panel_content = Text(content_to_show)
        
        # Escape brackets in path to prevent Rich markup parsing issues
        escaped_path = str(result.source).replace('[', r'\[').replace(']', r'\]')
        metadata_str = f"[dim]Path: {escaped_path}\nModified: {result.modified}[/dim]"
        
        panel = Panel(
            panel_content,
            title=title,
            subtitle=metadata_str,
            expand=False,
            border_style="blue"
        )
        
        console.print(panel)
        console.print()
    
    def _get_lexer_for_extension(self, ext: str) -> Optional[str]:
        lexer_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'objective-c',
            '.pl': 'perl',
            '.sh': 'bash',
            '.bash': 'bash',
            '.zsh': 'bash',
            '.fish': 'fish',
            '.ps1': 'powershell',
            '.sql': 'sql',
            '.html': 'html',
            '.htm': 'html',
            '.xml': 'xml',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'ini',
            '.conf': 'nginx',
            '.md': 'markdown',
            '.rst': 'rst',
            '.tex': 'latex',
            '.dockerfile': 'docker',
            '.makefile': 'makefile',
            '.cmake': 'cmake',
            '.vim': 'vim',
            '.lua': 'lua',
            '.dart': 'dart',
            '.elm': 'elm',
            '.clj': 'clojure',
            '.erl': 'erlang',
            '.ex': 'elixir',
            '.exs': 'elixir',
            '.fs': 'fsharp',
            '.ml': 'ocaml',
            '.pas': 'pascal',
            '.pp': 'pascal',
            '.d': 'd',
            '.zig': 'zig',
            '.v': 'verilog',
            '.vhd': 'vhdl',
            '.asm': 'nasm',
            '.s': 'gas',
        }
        return lexer_map.get(ext)