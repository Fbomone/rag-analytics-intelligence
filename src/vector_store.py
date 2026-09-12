# src/vector_store.py
import os
from typing import List, Dict

class VectorStore:
    """Simplified vector store without external dependencies"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.documents = []
        os.makedirs(persist_directory, exist_ok=True)
        print("✅ Vector Store initialized")
    
    def create_from_texts(self, texts: List[str], metadatas: List[dict] = None):
        """Store texts in memory"""
        self.documents = texts
        print(f"✅ Stored {len(texts)} documents")
        return self
    
    def load_existing(self):
        """Load from memory"""
        return self
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Simple search - returns first k documents"""
        return self.documents[:k] if self.documents else []
    
    def get_retriever(self, k: int = 3):
        """Return self as retriever"""
        return self
    
    def get_relevant_documents(self, query: str):
        """For retriever interface"""
        class DocResult:
            def __init__(self, content):
                self.page_content = content
        
        return [DocResult(doc) for doc in self.search(query)]