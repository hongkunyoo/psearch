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
    
    def _load_documents(self) -> List[Document]:
        documents = []
        extensions = ['.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.sh', '.sql']
        
        if not self.notes_dir.exists():
            console.print(f"[yellow]Notes directory {self.notes_dir} does not exist[/yellow]")
            return documents
        
        files = []
        for ext in extensions:
            files.extend(self.notes_dir.rglob(f"*{ext}"))
        
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