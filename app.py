import streamlit as st
import os
from src.SemanticDocumentProcessor import SemanticDocumentProcessor
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
    return SemanticDocumentProcessor(), VectorStore(), RAGPipeline()


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
