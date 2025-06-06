import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Optional OpenAI settings (for future upgrade)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vectorstore")

    # Document processing settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150

    # Retrieval settings
    TOP_K_RETRIEVAL = 3  # Fewer docs for local processing

    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LOCAL_LLM_MODEL = "microsoft/DialoGPT-medium"  # Free local model
    TEMPERATURE = 0.1
