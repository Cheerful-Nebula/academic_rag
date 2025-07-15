import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()


class Config:
    # Chroma settings
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vectorstore")

    # Document processing settings
    MIN_CHUNK_SIZE = 100  # Minimum chunk size in characters
    MAX_CHUNK_SIZE = 1000  # Maximum chunk size in characters
    MAX_SENTENCE_LENGTH = 200  # Maximum sentence length in characters
    MAX_PARAGRAPH_LENGTH = 500  # Maximum paragraph length in characters
    CHUNK_OVERLAP = 150  # Chunk overlap in characters
    OVERLAP_SENTENCES = 2  # Overlap sentences for sliding window
    SIMILARITY_THRESHOLD = 0.7  # Similarity threshold for chunk merging

    # Retrieval settings
    TOP_K_RETRIEVAL = 3  # Fewer docs for local processing

    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "google/flan-t5-base"  # "google/flan-t5-base"  "microsoft/DialoGPT-medium"  Free local model
    LLM_TEMPERATURE = 0.3  # Temperature for text generation
    EMBEDDING_MODEL_LOCAL_PATH = Path("models") / EMBEDDING_MODEL
