# app.py
import streamlit as st
from src.data_loader import DataLoader
from src.rag_engine import RAGAnalyticsEngine

# PAGE CONFIG
st.set_page_config(
    page_title="📊 RAG Analytics Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# TITLE
st.title("🤖 AI-Powered Analytics Intelligence Engine")
st.markdown("*Haz preguntas sobre tus datos en lenguaje natural*")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Configuración")
    st.info("""
    **RAG Analytics Engine** usa Cohere AI para analizar tus datos.
    
    **Stack:**
    - LLM: Cohere Command AI (command-a-plus-05-2026)
    - Vector DB: Chroma
    - Frontend: Streamlit
    """)

# INITIALIZE ENGINE
if "engine" not in st.session_state:
    with st.spinner("🔄 Inicializando..."):
        try:
            loader = DataLoader("./data")
            documents = loader.load_documents()
            
            engine = RAGAnalyticsEngine()
            engine.initialize_with_documents(documents)
            
            st.session_state.engine = engine
            st.success("✅ Motor IA listo!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# MAIN CONTENT
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("💬 Chat")
    
    # Show messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    st.markdown("---")
    
    # Input
    user_question = st.text_area(
        "Tu pregunta:",
        placeholder="¿Cuáles son nuestros mejores productos?",
        height=100
    )
    
    # Buttons
    col_btn1, col_btn2 = st.columns([2, 1])
    
    with col_btn1:
        if st.button("🚀 Obtener Insight", use_container_width=True, type="primary"):
            if user_question:
                st.session_state.messages.append({"role": "user", "content": user_question})
                
                with st.spinner("🧠 Analizando..."):
                    try:
                        response = st.session_state.engine.query(user_question)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Escribe una pregunta")
    
    with col_btn2:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.messages = []
            if "engine" in st.session_state:
                st.session_state.engine.clear_history()
            st.rerun()

with col2:
    st.subheader("📊 Datos")
    
    try:
        loader = DataLoader("./data")
        df = loader.load_csv("datasets/sample_data.csv")
        
        st.markdown("**Preview:**")
        st.dataframe(df.head(), use_container_width=True)
        
        st.metric("Total registros", len(df))
        
        if "sales" in df.columns:
            st.metric("Ventas promedio", f"${df['sales'].mean():,.0f}")
    except Exception as e:
        st.warning(f"⚠️ Error: {str(e)}")

# FOOTER
st.divider()
st.markdown("""
**Franco Bomone** | AI Analytics Engineer  
Stack: Python | Claude API | Streamlit | Chroma
""")