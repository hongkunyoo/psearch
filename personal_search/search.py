from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import re

from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
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
        
        # Initialize local LLM for relevance scoring
        self._init_llm()
    
    def _init_llm(self):
        """Initialize a local LLM for relevance scoring"""
        try:
            # Use a small, fast model for relevance scoring
            # Using FLAN-T5 small model which is good for Q&A and reasoning tasks
            model_name = "google/flan-t5-small"
            
            with console.status("[bold green]Loading LLM model for intelligent search..."):
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                # Create a pipeline for text generation
                self.llm_pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=50,
                    device=-1  # Use CPU
                )
                
                self.llm = HuggingFacePipeline(pipeline=self.llm_pipeline)
                
                # Create prompt template for relevance scoring
                self.relevance_prompt = PromptTemplate(
                    input_variables=["query", "filename", "content_preview"],
                    template="""Given the search query: "{query}"
                    
Is this file relevant? Answer with just 'yes' or 'no'.

File: {filename}
Content preview: {content_preview}

Answer:"""
                )
                
                self.score_prompt = PromptTemplate(
                    input_variables=["query", "filename", "content_preview"],
                    template="""Query: {query}
File name: {filename}
Content: {content_preview}

How relevant is this file to the query? Rate 1-10:"""
                )
                
            console.print("[green]✓ LLM model loaded successfully[/green]")
            self.llm_available = True
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load LLM model: {e}[/yellow]")
            console.print("[yellow]Falling back to keyword-based search[/yellow]")
            self.llm_available = False
    
    def _score_with_llm(self, result: SearchResult, query: str) -> float:
        """Use LLM to score the relevance of a result"""
        if not self.llm_available:
            return result.score
        
        try:
            # Prepare content preview (first 300 chars)
            content_preview = result.content[:300] if len(result.content) > 300 else result.content
            
            # Get relevance score from LLM using the new invoke method
            score_chain = self.score_prompt | self.llm
            
            llm_response = score_chain.invoke({
                "query": query,
                "filename": result.filename,
                "content_preview": content_preview
            }).strip()
            
            # Try to extract numeric score
            try:
                # Extract first number from response
                import re
                numbers = re.findall(r'\d+', llm_response)
                if numbers:
                    llm_score = int(numbers[0])
                    llm_score = max(1, min(10, llm_score))  # Clamp to 1-10
                else:
                    llm_score = 5  # Default middle score
            except:
                llm_score = 5
            
            # Combine with original score (lower is better)
            # Convert LLM score to distance-like metric
            llm_distance = (11 - llm_score) / 10.0  # Convert 1-10 to 1.0-0.1
            
            # Weight: 70% LLM score, 30% vector similarity
            combined_score = (llm_distance * 0.7) + (result.score * 0.3)
            
            return combined_score
            
        except Exception as e:
            console.print(f"[yellow]LLM scoring error: {e}[/yellow]", markup=True)
            return result.score
    
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
            # Get more results initially for better re-ranking
            initial_k = min(top_k * 2, 20) if self.llm_available else top_k
            
            # Search with original query
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=initial_k,
                filter=filter_dict
            )
            
            # Create SearchResult objects
            search_results = []
            seen_files = {}
            
            for doc, score in results_with_scores:
                file_path = str(doc.metadata.get('source', 'unknown'))
                
                # Skip results from Git directories
                if '/.git/' in file_path or '/.github/' in file_path:
                    continue
                
                # Deduplicate by file path
                if file_path not in seen_files:
                    result = SearchResult(doc, score)
                    seen_files[file_path] = result
                    search_results.append(result)
            
            # Apply LLM-based scoring if available
            if self.llm_available and search_results:
                console.print(f"[cyan]Analyzing {len(search_results)} results with LLM...[/cyan]")
                
                for result in search_results:
                    # Score each result with LLM
                    result.score = self._score_with_llm(result, query)
            
            # Sort by score (lower is better) and return top_k
            search_results.sort(key=lambda x: x.score)
            
            return search_results[:top_k]
            
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