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
        self.df = None  # Para datos CSV
        
        print("✅ RAG Analytics Engine initialized with Cohere")
    
    def initialize_with_documents(self, documents: list):
        """Store documents"""
        self.documents = documents
        print(f"✅ Loaded {len(documents)} documents")
    
    def set_dataframe(self, df):
        """Set DataFrame con datos CSV"""
        self.df = df
        print(f"✅ Loaded {len(df)} rows from CSV")
    
    def _get_data_summary(self) -> str:
        """Genera resumen de datos CSV"""
        if self.df is None:
            return ""
        
        try:
            summary = "\n📊 DATOS DISPONIBLES:\n"
            summary += "=" * 50 + "\n"
            summary += f"Total registros: {len(self.df)}\n"
            summary += f"Columnas: {', '.join(self.df.columns.tolist())}\n\n"
            
            # Resumen por columna numérica
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                summary += f"{col}:\n"
                summary += f"  - Total: {self.df[col].sum():.0f}\n"
                summary += f"  - Promedio: {self.df[col].mean():.1f}\n"
                summary += f"  - Min: {self.df[col].min():.0f}\n"
                summary += f"  - Max: {self.df[col].max():.0f}\n"
            
            # Mostrar algunos registros
            summary += "\nPrimeros 5 registros:\n"
            summary += self.df.head().to_string()
            
            return summary
        except Exception as e:
            return f"Error al procesar datos: {e}"
    
    def query(self, question: str) -> str:
        """Query con datos CSV y documentos"""
    
        print(f"\n🔍 Pregunta: {question}")
    
       # Build context
        context = "📄 CONTEXTO:\n"
    context += "=" * 50 + "\n"
    
    # Agregar datos CSV si existen
    if self.df is not None:
        context += self._get_data_summary()
        context += "\n\n"
    
    # Agregar documentos si existen
    if self.documents:
        context += "📋 DOCUMENTOS RELEVANTES:\n"
        for i, doc in enumerate(self.documents[:2], 1):
            content = doc.get('content', '')[:300]
            context += f"{i}. {content}\n"
    
    user_message = f"""{context}

Pregunta: {question}

Análisis:"""
    
    print("🤖 Cohere analizando...")
    
    try:
        # Convertir chat_history al formato v2
        messages = []
        for msg in self.conversation_history:
            if msg["role"] == "USER":
                messages.append({"role": "user", "content": msg["message"]})
            else:
                messages.append({"role": "assistant", "content": msg["message"]})
        
        # Agregar mensaje actual
        messages.append({"role": "user", "content": user_message})
        
        # Usar API v2 correctamente
        response = self.client.chat(
            model="command-a-plus-05-2026",
            messages=messages
        )
        
        assistant_response = response.message.content[0].text
        
        # Guardar en historial
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
        print(f"Error: {e}")
        return f"❌ Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation"""
        self.conversation_history = []
        print("🗑️  Chat limpiado")