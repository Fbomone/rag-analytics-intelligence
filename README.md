# RAG Analytics Intelligence Engine

AI-powered analytics system using Claude AI.

## 🎯 Features
- Retrieval-Augmented Generation (RAG)
- Natural language queries
- Real-time insights
- Streamlit UI

## 🚀 Quick Start

### Requirements
- Python 3.9+
- Claude API key (free at https://www.anthropic.com/api)

### Installation

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure .env with your API keys
# Then run:
streamlit run app.py
```

Open http://localhost:8501

## 📊 Tech Stack
- **LLM:** Claude 3.5 Sonnet
- **Vector DB:** Chroma
- **Frontend:** Streamlit
- **Embeddings:** HuggingFace

## 📁 Project Structure