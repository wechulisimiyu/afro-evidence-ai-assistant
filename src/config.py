"""
Configuration settings for the application.
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from loguru import logger

# Load environment variables
load_dotenv()

# API Settings
APP_HOST = os.getenv("APP_HOST", "localhost")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_DEBUG = os.getenv("APP_DEBUG", "False").lower() == "true"

# HMIS API Settings
HMIS_API_URL = os.getenv("HMIS_API_URL", "http://localhost:8001")
HMIS_API_KEY = os.getenv("HMIS_API_KEY")
HMIS_API_ENABLED = os.getenv("HMIS_API_ENABLED", "False").lower() == "true"

# Google AI Settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Model Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# Model Generation Settings
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "30720"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
TOP_K = int(os.getenv("TOP_K", "40"))

# Rate Limiting Settings
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "60"))
TOKENS_PER_MINUTE = int(os.getenv("TOKENS_PER_MINUTE", "1000000"))

# Vector Store Settings
GUIDELINES_VECTOR_STORE = os.path.join("src", "data", "guidelines_vector_store")
JOURNALS_VECTOR_STORE = os.path.join("src", "data", "journals_vector_store")

# Configure Gemini model
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(
    model_name=LLM_MODEL,
    generation_config={
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
    }
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# Initialize vector stores
try:
    guidelines_store = FAISS.load_local(
        GUIDELINES_VECTOR_STORE, embeddings, allow_dangerous_deserialization=True
    )
    logger.info("Guidelines vector store loaded successfully")
except Exception as e:
    logger.error(f"Error loading guidelines store: {str(e)}")
    guidelines_store = None

try:
    journals_store = FAISS.load_local(
        JOURNALS_VECTOR_STORE, embeddings, allow_dangerous_deserialization=True
    )
    logger.info("Journals vector store loaded successfully")
except Exception as e:
    logger.error(f"Error loading journals store: {str(e)}")
    journals_store = None

# Logging Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
