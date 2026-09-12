# src/rag_engine.py
import os
from dotenv import load_dotenv

load_dotenv()

class RAGAnalyticsEngine:
    def __init__(self, vector_db_path: str = "./chroma_db"):
        """Initialize RAG Engine with Cohere"""
        import cohere
        
        self.api_key = os.getenv("COHERE_API_KEY")
        self.client = cohere.Client(api_key=self.api_key)
        self.conversation_history = []
        self.documents = []
        
        print("✅ RAG Analytics Engine initialized with Cohere")
    
    def initialize_with_documents(self, documents: list):
        """Store documents"""
        self.documents = documents
        print(f"✅ Loaded {len(documents)} documents")
    
    def query(self, question: str) -> str:
        """Query with RAG using Cohere"""
        
        print(f"\n🔍 Pregunta: {question}")
        
        # Build context from documents
        context = "📄 Documentos:\n"
        if self.documents:
            for i, doc in enumerate(self.documents[:2], 1):
                content = doc.get('content', '')[:300]
                context += f"{i}. {content}\n"
        
        user_message = f"""{context}

Pregunta: {question}

Responde en español, de forma concisa."""
        
        print("🤖 Cohere analizando...")
        
        try:
            response = self.client.chat(
                message=user_message,
                chat_history=self.conversation_history,
                model="command-a-plus-05-2026"
            )
            
            assistant_response = response.text
            
            self.conversation_history.append({
                "role": "USER",
                "message": question
            })
            
            self.conversation_history.append({
                "role": "CHATBOT",
                "message": assistant_response
            })
            
            print("✅ Análisis completado")
            return assistant_response
        
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation"""
        self.conversation_history = []
        print("🗑️  Chat limpiado")