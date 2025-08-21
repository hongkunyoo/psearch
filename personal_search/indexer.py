import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import settings

console = Console()


class NotesIndexer:
    def __init__(
        self,
        notes_dir: Optional[Path] = None,
        index_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
    ):
        self.notes_dir = notes_dir or settings.notes_directory
        self.index_dir = index_dir or settings.index_directory
        self.embedding_model = embedding_model or settings.embedding_model
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        if settings.use_local_embeddings:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        
        self.vectorstore = None
    
    def _get_file_hash(self, filepath: Path) -> str:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _should_skip_path(self, filepath: Path) -> bool:
        """Check if a path should be skipped during indexing"""
        # Directories to skip
        skip_dirs = {'.git', '.github', '__pycache__', 'node_modules', '.venv', 'venv', 'env'}
        
        # Check if any parent directory should be skipped
        for parent in filepath.parents:
            if parent.name in skip_dirs:
                return True
        
        # Check if the file itself is in a skip directory
        if filepath.parent.name in skip_dirs:
            return True
            
        # Check for .git and .github anywhere in the path
        path_str = str(filepath)
        if '/.git/' in path_str or '/.github/' in path_str:
            return True
            
        return False
    
    def _is_text_file(self, filepath: Path) -> bool:
        """Check if a file is likely to be a text file"""
        try:
            with open(filepath, 'rb') as f:
                # Read first 1024 bytes to check for binary content
                chunk = f.read(1024)
                if not chunk:
                    return True  # Empty files are considered text
                
                # Check for null bytes (common in binary files)
                if b'\x00' in chunk:
                    return False
                
                # Try to decode as UTF-8
                try:
                    chunk.decode('utf-8')
                    return True
                except UnicodeDecodeError:
                    pass
                
                # Try to decode as latin-1 (fallback)
                try:
                    chunk.decode('latin-1')
                    return True
                except UnicodeDecodeError:
                    return False
                    
        except (OSError, IOError):
            return False

    def _load_documents(self) -> List[Document]:
        documents = []
        extensions = ['.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.sh', '.sql']
        
        if not self.notes_dir.exists():
            console.print(f"[yellow]Notes directory {self.notes_dir} does not exist[/yellow]")
            return documents
        
        files = []
        # Add files with known extensions
        for ext in extensions:
            potential_files = self.notes_dir.rglob(f"*{ext}")
            for filepath in potential_files:
                if not self._should_skip_path(filepath):
                    files.append(filepath)
        
        # Add extension-less files that appear to be text files
        all_files = self.notes_dir.rglob("*")
        for filepath in all_files:
            if (filepath.is_file() and 
                not filepath.suffix and 
                not self._should_skip_path(filepath) and 
                self._is_text_file(filepath)):
                files.append(filepath)
        
        # Remove duplicates while preserving order
        files = list(dict.fromkeys(files))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Loading {len(files)} files...", total=len(files))
            
            for filepath in files:
                try:
                    loader = TextLoader(str(filepath), encoding='utf-8')
                    file_docs = loader.load()
                    
                    for doc in file_docs:
                        # Include filename in the content for better search relevance
                        # Format: "Filename: <name>\n<original content>"
                        filename_header = f"Filename: {filepath.name}\nPath: {filepath.parent.name}/{filepath.name}\n\n"
                        doc.page_content = filename_header + doc.page_content
                        
                        doc.metadata.update({
                            'source': str(filepath),
                            'filename': filepath.name,
                            'modified': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                            'file_hash': self._get_file_hash(filepath),
                        })
                    
                    documents.extend(file_docs)
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    console.print(f"[red]Error loading {filepath}: {e}[/red]")
                    progress.update(task, advance=1)
        
        return documents
    
    def index(self, force_reindex: bool = False) -> int:
        console.print(f"[bold blue]Indexing notes from {self.notes_dir}[/bold blue]")
        
        documents = self._load_documents()
        
        if not documents:
            console.print("[yellow]No documents found to index[/yellow]")
            return 0
        
        console.print(f"[green]Loaded {len(documents)} documents[/green]")
        
        chunks = self.text_splitter.split_documents(documents)
        console.print(f"[green]Split into {len(chunks)} chunks[/green]")
        
        persist_directory = str(self.index_dir)
        
        if force_reindex and self.index_dir.exists():
            import shutil
            shutil.rmtree(self.index_dir)
            self.index_dir.mkdir(parents=True, exist_ok=True)
        
        with console.status("[bold green]Creating vector embeddings..."):
            if self.index_dir.exists() and any(self.index_dir.iterdir()):
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings,
                )
                self.vectorstore.add_documents(chunks)
            else:
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=persist_directory,
                )
        
        console.print(f"[bold green]✓ Indexed {len(chunks)} chunks successfully![/bold green]")
        return len(chunks)
    
    def get_vectorstore(self) -> Optional[Chroma]:
        if self.vectorstore is None:
            persist_directory = str(self.index_dir)
            if self.index_dir.exists() and any(self.index_dir.iterdir()):
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings,
                )
        return self.vectorstore