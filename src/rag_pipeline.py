from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import torch # noqa
from .vector_store import VectorStore
from .config import Config # noqa


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
            f"Source {i+1}: {doc['content']}" if len(doc['content']) <= 500
            else f"Source {i+1}: {doc['content'][:500]}..."  # Truncate for efficiency

            for i, doc in enumerate(context_docs[:5])  # Use top 5 docs only
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
