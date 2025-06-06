# Academic Research Assistant RAG - Phase 1 Implementation

## Project Structure
```
academic-rag/
├── requirements.txt
├── .env
├── app.py                    # Main Streamlit app
├── src/
│   ├── __init__.py
│   ├── document_processor.py # PDF processing & chunking
│   ├── vector_store.py       # Chroma operations
│   ├── rag_pipeline.py       # Core RAG logic
│   └── config.py            # Configuration settings
├── data/
│   ├── raw/                 # Upload folder for PDFs
│   └── processed/           # Cleaned text files
├── vectorstore/             # Chroma database files
└── README.md
```

## 1. requirements.txt (Free Version)
```txt
streamlit==1.29.0
langchain==0.1.0
chromadb==0.4.18
pypdf2==3.0.1
sentence-transformers==2.2.2
python-dotenv==1.0.0
tiktoken==0.5.2
pandas==2.1.4
numpy==1.24.3
transformers==4.36.0
torch==2.1.0
accelerate==0.25.0
```

## 2. .env (Free Version - Optional)
```bash
# Optional: Only needed if you decide to use OpenAI later
# OPENAI_API_KEY=your_openai_api_key_here
CHROMA_PERSIST_DIRECTORY=./vectorstore
```

## 3. src/config.py (Free Version)
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Optional OpenAI settings (for future upgrade)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vectorstore")
    
    # Document processing settings
    CHUNK_SIZE = 800  # Smaller chunks for local models
    CHUNK_OVERLAP = 100
    
    # Retrieval settings
    TOP_K_RETRIEVAL = 3  # Fewer docs for local processing
    
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LOCAL_LLM_MODEL = "microsoft/DialoGPT-medium"  # Free local model
    TEMPERATURE = 0.1
```

## 4. src/document_processor.py
```python
import PyPDF2
import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import Config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
    
    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "metadata": {
                    "source": source,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            })
        
        return documents
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Complete PDF processing pipeline"""
        filename = os.path.basename(pdf_path)
        text = self.extract_text_from_pdf(pdf_path)
        documents = self.chunk_text(text, filename)
        
        return documents
```

## 5. src/vector_store.py
```python
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .config import Config
import os

class VectorStore:
    def __init__(self):
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIRECTORY)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="academic_papers",
            metadata={"description": "Academic research papers collection"}
        )
    
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to vector store"""
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Generate IDs
        ids = [f"{doc['metadata']['source']}_{doc['metadata']['chunk_id']}" 
               for doc in documents]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query: str, top_k: int = Config.TOP_K_RETRIEVAL) -> List[Dict]:
        """Search for relevant documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "embedding_model": Config.EMBEDDING_MODEL
        }
```

## 6. src/rag_pipeline.py (Free Version with Hugging Face)
```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import torch
from .vector_store import VectorStore
from .config import Config

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        
        # Initialize free local model
        print("Loading language model... (this may take a few minutes first time)")
        
        # Use a smaller, efficient model that runs locally
        model_name = "microsoft/DialoGPT-medium"  # or "google/flan-t5-base"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Set up the pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("✅ Language model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to a simpler approach
            self.generator = None
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> Dict:
        """Generate response using retrieved context"""
        
        if not self.generator:
            return {
                "answer": "Language model not available. Please check your setup.",
                "sources": [],
                "context_used": 0
            }
        
        # Prepare context (keep it shorter for local models)
        context = "\n".join([
            f"Source {i+1}: {doc['content'][:300]}..."  # Truncate for efficiency
            for i, doc in enumerate(context_docs[:3])  # Use top 3 docs only
        ])
        
        # Create a focused prompt
        prompt = f"""Based on these research excerpts, answer the question:

Context: {context}

Question: {query}
Answer:"""

        try:
            # Generate response
            response = self.generator(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.1
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Get just the answer part (after "Answer:")
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            return {
                "answer": answer,
                "sources": [doc['metadata']['source'] for doc in context_docs],
                "context_used": len(context_docs)
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "context_used": 0
            }
    
    def query(self, question: str) -> Dict:
        """Complete RAG pipeline"""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(question)
        
        if not relevant_docs:
            return {
                "answer": "No relevant documents found in the database.",
                "sources": [],
                "context_used": 0
            }
        
        # Generate response
        response = self.generate_response(question, relevant_docs)
        
        return response

# Alternative: OpenAI version (uncomment if you get API key)
"""
from openai import OpenAI

class RAGPipelineOpenAI:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.vector_store = VectorStore()
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> Dict:
        # [Previous OpenAI implementation]
        pass
"""
```

## 7. app.py (Main Streamlit App)
```python
import streamlit as st
import os
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="Academic Research Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize components
@st.cache_resource
def init_components():
    return DocumentProcessor(), VectorStore(), RAGPipeline()

processor, vector_store, rag = init_components()

# Main app
st.title("📚 Academic Research Assistant")
st.markdown("Upload academic papers and ask questions about their content!")

# Sidebar for document management
with st.sidebar:
    st.header("Document Management")
    
    # Collection stats
    stats = vector_store.get_collection_stats()
    st.metric("Documents in Database", stats["total_documents"])
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file
                    os.makedirs("data/raw", exist_ok=True)
                    file_path = f"data/raw/{uploaded_file.name}"
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Process and add to vector store
                    try:
                        documents = processor.process_pdf(file_path)
                        vector_store.add_documents(documents)
                        st.success(f"✅ Processed {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                
                st.rerun()

# Main query interface
st.header("Ask Questions")

# Query input
question = st.text_input(
    "What would you like to know about your documents?",
    placeholder="e.g., What are the main findings about machine learning in education?"
)

if question:
    with st.spinner("Searching and generating response..."):
        result = rag.query(question)
        
        # Display answer
        st.subheader("📝 Answer")
        st.write(result["answer"])
        
        # Display sources
        if result["sources"]:
            st.subheader("📖 Sources")
            for source in set(result["sources"]):
                st.write(f"• {source}")
        
        # Display metadata
        with st.expander("🔍 Query Details"):
            st.write(f"**Context documents used:** {result['context_used']}")

# Example questions
with st.expander("💡 Example Questions"):
    st.write("""
    - What are the main methodologies discussed in these papers?
    - What are the key findings about [specific topic]?
    - How do the authors define [specific concept]?
    - What are the limitations mentioned in the research?
    - What future work do the authors suggest?
    """)
```

## 8. README.md Structure
```markdown
# Academic Research Assistant RAG System

A retrieval-augmented generation system for querying academic research papers.

## Features
- PDF document processing and chunking
- Vector similarity search using sentence transformers
- OpenAI GPT integration for response generation
- Interactive Streamlit interface
- Source attribution and citation

## Quick Start
1. Clone repository
2. Install requirements: `pip install -r requirements.txt`
3. Set up environment variables in `.env`
4. Run: `streamlit run app.py`

## Usage
1. Upload PDF research papers
2. Ask questions about the content
3. Get AI-powered answers with source citations

## Demo
[Add screenshots and demo GIF here]
```

## Next Steps (Week 1-2)
1. **Day 1-2**: Set up project structure and basic PDF processing
2. **Day 3-4**: Implement vector store functionality
3. **Day 5-7**: Build RAG pipeline and test with sample papers
4. **Week 2**: Polish UI, add error handling, create demo

## Testing Strategy
- Start with 3-5 research papers from ArXiv
- Test with various question types (factual, comparative, analytical)
- Verify source attribution works correctly
- Check performance with different chunk sizes

This structure gives you a complete, working RAG system that demonstrates all the key concepts employers look for. Ready to start coding?