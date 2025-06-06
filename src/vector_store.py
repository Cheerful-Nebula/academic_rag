import chromadb
from chromadb.config import Settings  # noqa
from sentence_transformers import SentenceTransformer
from config import Config


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

    def add_documents(self, documents: list[dict]) -> None:
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

    def search(self, query: str, top_k: int = Config.TOP_K_RETRIEVAL) -> list[dict]:
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

    def get_collection_stats(self) -> dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "embedding_model": Config.EMBEDDING_MODEL
        }
