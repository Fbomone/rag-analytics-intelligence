# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# ===== API KEYS =====
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ===== MODEL CONFIG =====
MODEL_NAME = "claude-3-5-sonnet-20241022"
TEMPERATURE = 0.7
MAX_TOKENS = 1000

# ===== VECTOR STORE =====
VECTOR_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ===== DATA PATHS =====
DATA_PATH = "./data"
SAMPLE_DATA_FILE = "datasets/sample_data.csv"
DOCUMENTS_PATH = "documents/"

print("✅ Configuration loaded")