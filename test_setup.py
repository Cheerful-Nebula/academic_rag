# test_setup.py - Run this to verify your FREE setup works


from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline


def test_environment():
    """Test if all components are working"""
    print("🔍 Testing Academic RAG Setup (FREE VERSION)...")

    # Test 1: Sentence Transformers
    try:
        print("📥 Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode(["This is a test sentence"])
        print(f"✅ Sentence Transformers working - embedding shape: {test_embedding.shape}")
    except Exception as e:
        print(f"❌ Sentence Transformers error: {e}")
        return False

    # Test 2: ChromaDB
    try:
        print("📊 Testing ChromaDB...")
        client = chromadb.PersistentClient(path="./test_vectorstore")
        collection = client.get_or_create_collection("test") # noqa
        print("✅ ChromaDB working")
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False

    # Test 3: Local Language Model
    try:
        print("🤖 Testing local language model (this may take a few minutes first time)...")

        # Use a simple, fast model for testing
        generator = pipeline(
            "text-generation",
            model="distilgpt2",  # Smaller model for testing
            max_length=50
        )

        test_response = generator("The capital of France is", max_new_tokens=10)
        print("✅ Local language model working")
        print(f"🧪 Test response: {test_response[0]['generated_text']}")

    except Exception as e:
        print(f"❌ Language model error: {e}")
        print("💡 This is normal - language models are large downloads")
        return False

    print("\n🎉 All tests passed! You're ready to build your FREE RAG system!")
    print("💰 No API costs - everything runs locally!")
    return True


if __name__ == "__main__":
    success = test_environment()

    if success:
        print("\n📚 Next steps:")
        print("1. Add some PDF files to data/raw/")
        print("2. Run: python test_document_processing.py")
        print("3. Run: streamlit run app.py")
    else:
        print("\n🔧 Fix the errors above, then try again")
