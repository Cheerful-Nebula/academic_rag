from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch # noqa
from .vector_store import VectorStore
from .config import Config # noqa
import re


class RAGPipeline:
    def __init__(self):
        # Use a smaller, efficient model that runs locally
        # Initialize model, tokenizer, genetor pipeline, and vector store
        model_name = "google/flan-t5-base"  # "google/flan-t5-base" "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
        self.vector_store = VectorStore()
        self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.3,
                pad_token_id=self.tokenizer.eos_token_id
            )
        # --- NEW: Structured Prompt Template ---
        # This template is designed for academic analysis tasks
        self.PROMPT_TEMPLATE = """
        You're an AI research assistant analyzing academic papers. Use the context to answer scientifically.

        **STRUCTURED CONTEXT:**
        {context}

        **ANALYSIS TASK:**
        1. Identify key claims relevant to: "{query}"
        2. Cross-reference across sources
        3. Generate evidence-based response
        4. Cite sources [1-3] where applicable

        **RESPONSE:**
        """

        print("✅ Language model loaded successfully!")

    def format_context(docs: list[dict]) -> str:
        context_lines = []
        for i, doc in enumerate(docs):
            metadata = doc['metadata']

            # PRESERVE STRUCTURE
            header = f"SOURCE {i+1} | {metadata['source']}"
            section = f"SECTION: {metadata['section']}" if 'section' in metadata else ""
            chunk_type = f"CHUNK TYPE: {metadata['chunk_type']}"

            # SMART TRUNCATION (preserve sentence boundaries)
            content = doc['content']
            if len(content) > 400:
                last_period = content[:400].rfind('.')
                content = content[:last_period+1] + " [...]"

            context_lines.append(f"{header}\n{section}\n{chunk_type}\n{content}\n")

        return "\n\n".join(context_lines)

    def postprocess(self, raw_response, context_docs):
        # Extract answer
        answer = raw_response[0]['generated_text'].split("RESPONSE:")[-1].strip()

        # Extract citations
        source_ids = set()
        for i in range(len(context_docs)):
            if f"[{i+1}]" in answer:
                source_ids.add(i)

        # Verify citations
        valid_sources = [d['metadata']['source'] for i, d in enumerate(context_docs) if i in source_ids]

        return {
            "answer": answer,
            "sources": valid_sources,
            "context_used": len(context_docs)
        }

    def generate_response(self, query: str, context_docs: list[dict]) -> dict:
        """Generate response using retrieved context"""

        if not self.generator:
            return {
                "answer": "Language model not available. Please check your setup.",
                "sources": [],
                "context_used": 0
            }

        # Prepare context (keep it shorter for local models)
        # STRUCTURED context formatting
        formatted_context = self.format_context(context_docs[:3])  # Use top 3 most relevant

        # DYNAMIC prompt construction
        prompt = self.PROMPT_TEMPLATE.format(context=formatted_context, query=query)

        try:
            # GENERATE with precision control
            response = self.generator(
                prompt,
                max_new_tokens=350,
                temperature=0.2,  # Lower for factual accuracy
                repetition_penalty=1.2,
                num_beams=5,      # Better than greedy search
                do_sample=False
            )

            # Extract the generated text
            generated_text = response[0]['generated_text']

            # --- NEW: Citation-Aware Postprocessing ---
            answer = self._extract_answer(generated_text)
            sources = self._extract_citations(answer, context_docs)

            return {
                "answer": answer,
                "sources": sources,  # Only includes cited sources!
                "context_used": len(context_docs),
                "citations": self._find_citation_indices(answer)  # Bonus: citation locations
            }

        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "context_used": 0
            }
    # --- NEW HELPER METHODS ---

    def _format_context(self, context_docs: list[dict]) -> str:
        """Formats semantic chunks with metadata for the prompt."""
        context_lines = []
        for i, doc in enumerate(context_docs):
            meta = doc['metadata']
            header = f"📄 SOURCE {i+1} | {meta.get('source', 'unknown')}"
            section = f"🔖 SECTION: {meta.get('section', 'N/A')}"
            content = doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
            context_lines.append(f"{header}\n{section}\n{content}\n")
        return "\n".join(context_lines)

    def _extract_answer(self, generated_text: str) -> str:
        """Extracts the answer after '[RESPONSE]', or returns full text if marker missing."""
        return (
            generated_text.split("[RESPONSE]")[-1].strip()
            if "[RESPONSE]" in generated_text
            else generated_text
        )

    def _extract_citations(self, answer: str, context_docs: list[dict]) -> list[str]:
        """Returns only sources actually cited in the answer (e.g., [1], [2])."""
        cited_sources = []
        for i, doc in enumerate(context_docs):
            if f"[{i+1}]" in answer:  # Checks for citations like [1], [2], etc.
                cited_sources.append(doc['metadata']['source'])
        return cited_sources

    def _find_citation_indices(self, answer: str) -> list[int]:
        """Optional: Returns positions of citations in the answer (for highlighting)."""
        return [m.start() for m in re.finditer(r'\[\d+\]', answer)]

    def query(self, question: str) -> dict:
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
# This code implements a Retrieval-Augmented Generation (RAG) pipeline
# that retrieves relevant documents from a vector store and generates a response
# using a language model. It includes structured context formatting,
# dynamic prompt construction, and citation-aware postprocessing.
#             context_lines.append(f"{header}\n{section}\n{chunk_type}\n{content}\n")
#         return "\n\n".join(context_lines)
#         """Extracts citations from the answer based on context documents."""
#         citations = []
#         for i, doc in enumerate(context_docs):
#             if f"[{i+1}]" in answer:
#                 citations.append(doc['metadata']['source'])
#         return citations
#         return citations
#         """Finds indices of citations in the answer text."""
