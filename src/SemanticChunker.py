from sentence_transformers import SentenceTransformer
from config import Config # noqa
import re
import spacy
from typing import Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    content: str
    start_pos: int
    end_pos: int
    chunk_type: str  # 'sentence', 'paragraph', 'section'
    section: Optional[str] = None
    metadata: dict = None


class SemanticChunker:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        # Load spaCy model for sentence segmentation
        # Customize spaCy pipeline for efficiency
        self.nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])
        self.nlp.enable_pipe("senter")  # Modern sentence boundary detection

        # Load embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(embedding_model)

        # Academic paper section patterns
        self.section_patterns = [
            r'^(abstract|introduction|background|related work|methodology|methods|approach|model|architecture|\
                experiments|results|evaluation|discussion|conclusion|references|acknowledgments?)\b',
            r'^\d+\.?\s+(introduction|background|methodology|results|discussion|conclusion)',
            r'^[IVX]+\.?\s+(introduction|background|methodology|results|discussion|conclusion)'
        ]

        # Configuration
        self.min_chunk_size = 100
        self.max_chunk_size = 1000
        self.overlap_sentences = 2
        self.similarity_threshold = 0.7

    def detect_sections(self, text: str) -> list[tuple[str, int, int]]:
        """Detect academic paper sections and their boundaries."""
        sections = []
        lines = text.split('\n')
        current_section = "introduction"  # default
        section_start = 0

        for i, line in enumerate(lines):
            line_clean = line.strip().lower()

            # Skip empty lines
            if not line_clean:
                continue

            # Check if line matches section pattern
            for pattern in self.section_patterns:
                if re.match(pattern, line_clean):
                    # Close previous section
                    if sections or current_section != "introduction":
                        section_end = sum(len(lines[j]) + 1 for j in range(section_start, i))
                        sections.append((current_section,
                                         sum(len(lines[j]) + 1 for j in range(section_start)),
                                         section_end))

                    # Start new section
                    current_section = re.match(pattern, line_clean).group(1)
                    section_start = i
                    break

        # Add final section
        if current_section:
            sections.append((current_section,
                             sum(len(lines[j]) + 1 for j in range(section_start)),
                             len(text)))

        return sections if sections else [("content", 0, len(text))]

    def split_into_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences with position tracking."""
        doc = self.nlp(text)
        sentences = []

        for sent in doc.sents:
            # Clean sentence text
            sent_text = sent.text.strip()
            if len(sent_text) > 20:  # Filter out very short sentences
                sentences.append((sent_text, sent.start_char, sent.end_char))

        return sentences

    def split_into_paragraphs(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into paragraphs."""
        paragraphs = []
        current_pos = 0

        # Split by double newlines (paragraph breaks)
        para_texts = re.split(r'\n\s*\n', text)

        for para_text in para_texts:
            para_text = para_text.strip()
            if len(para_text) > 50:  # Filter short paragraphs
                start_pos = text.find(para_text, current_pos)
                if start_pos != -1:
                    end_pos = start_pos + len(para_text)
                    paragraphs.append((para_text, start_pos, end_pos))
                    current_pos = end_pos

        return paragraphs

    def calculate_semantic_similarity(self, texts: list[str]) -> np.ndarray:
        """Calculate semantic similarity matrix for texts."""
        if len(texts) < 2:
            return np.array([[1.0]])

        embeddings = self.embedding_model.encode(texts)
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

    def merge_similar_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge semantically similar adjacent chunks."""
        if len(chunks) <= 1:
            return chunks

        # Get chunk texts for similarity calculation
        chunk_texts = [chunk.content for chunk in chunks]
        similarity_matrix = self.calculate_semantic_similarity(chunk_texts)

        merged_chunks = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]

            # Check if we can merge with next chunk
            if (i + 1 < len(chunks) and
                len(current_chunk.content) < self.max_chunk_size and
                    similarity_matrix[i][i + 1] > self.similarity_threshold):

                next_chunk = chunks[i + 1]

                # Merge chunks
                merged_content = current_chunk.content + "\n\n" + next_chunk.content
                if len(merged_content) <= self.max_chunk_size:
                    merged_chunk = Chunk(
                        content=merged_content,
                        start_pos=current_chunk.start_pos,
                        end_pos=next_chunk.end_pos,
                        chunk_type="merged_paragraph",
                        section=current_chunk.section,
                        metadata={
                            "merged_from": [current_chunk.chunk_type, next_chunk.chunk_type],
                            "semantic_similarity": similarity_matrix[i][i + 1]
                        }
                    )
                    merged_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk since it's merged
                    continue

            merged_chunks.append(current_chunk)
            i += 1

        return merged_chunks

    def create_sliding_window_chunks(self, sentences: list[tuple[str, int, int]], section: str) -> list[Chunk]:
        """Create overlapping chunks with semantic awareness."""
        chunks = []
        i = 0

        while i < len(sentences):
            chunk_sentences = []
            chunk_length = 0
            start_idx = i

            # Build chunk up to max size
            while (i < len(sentences) and
                   chunk_length + len(sentences[i][0]) < self.max_chunk_size):
                chunk_sentences.append(sentences[i])
                chunk_length += len(sentences[i][0])
                i += 1

            if chunk_sentences:
                chunk_content = " ".join([sent[0] for sent in chunk_sentences])

                chunk = Chunk(
                    content=chunk_content,
                    start_pos=chunk_sentences[0][1],
                    end_pos=chunk_sentences[-1][2],
                    chunk_type="sliding_window",
                    section=section,
                    metadata={
                        "sentence_count": len(chunk_sentences),
                        "start_sentence_idx": start_idx,
                        "end_sentence_idx": i - 1
                    }
                )
                chunks.append(chunk)

                # Move back for overlap (but ensure progress)
                overlap_back = min(self.overlap_sentences, len(chunk_sentences) - 1, i - start_idx - 1)
                i = max(start_idx + 1, i - overlap_back)

        return chunks

    def chunk_text_semantically(self, text: str, source: str) -> list[dict]:
        """Main semantic chunking method."""
        # Step 1: Detect sections
        sections = self.detect_sections(text)

        all_chunks = []

        for section_name, section_start, section_end in sections:
            section_text = text[section_start:section_end]

            # Step 2: Split into paragraphs
            paragraphs = self.split_into_paragraphs(section_text)

            # Step 3: Create paragraph-level chunks
            paragraph_chunks = []
            for para_text, para_start, para_end in paragraphs:
                if len(para_text) >= self.min_chunk_size:
                    chunk = Chunk(
                        content=para_text,
                        start_pos=section_start + para_start,
                        end_pos=section_start + para_end,
                        chunk_type="paragraph",
                        section=section_name,
                        metadata={"paragraph_length": len(para_text)}
                    )
                    paragraph_chunks.append(chunk)
                elif len(para_text) > 20:  # Small paragraphs - will be merged
                    chunk = Chunk(
                        content=para_text,
                        start_pos=section_start + para_start,
                        end_pos=section_start + para_end,
                        chunk_type="small_paragraph",
                        section=section_name,
                        metadata={"paragraph_length": len(para_text)}
                    )
                    paragraph_chunks.append(chunk)

            # Step 4: Merge small paragraphs
            paragraph_chunks = self.merge_similar_chunks(paragraph_chunks)

            # Step 5: Handle large paragraphs with sliding window
            section_chunks = []
            for chunk in paragraph_chunks:
                if len(chunk.content) > self.max_chunk_size:
                    # Split large paragraph into sentences and use sliding window
                    sentences = self.split_into_sentences(chunk.content)
                    sliding_chunks = self.create_sliding_window_chunks(sentences, section_name)
                    section_chunks.extend(sliding_chunks)
                else:
                    section_chunks.append(chunk)

            all_chunks.extend(section_chunks)

        # Step 6: Create hierarchical chunks (section-level)
        hierarchical_chunks = []
        for section_name, section_start, section_end in sections:
            section_text = text[section_start:section_end]
            if len(section_text) > self.min_chunk_size:
                section_chunk = Chunk(
                    content=section_text,
                    start_pos=section_start,
                    end_pos=section_end,
                    chunk_type="section",
                    section=section_name,
                    metadata={
                        "section_length": len(section_text),
                        "is_hierarchical": True
                    }
                )
                hierarchical_chunks.append(section_chunk)

        # Combine all chunks
        all_chunks.extend(hierarchical_chunks)

        # Step 7: Convert to output format
        documents = []
        for i, chunk in enumerate(all_chunks):
            documents.append({
                "content": chunk.content,
                "metadata": {
                    "source": source,
                    "chunk_id": i,
                    "chunk_type": chunk.chunk_type,
                    "section": chunk.section,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                    "total_chunks": len(all_chunks),
                    **(chunk.metadata or {})
                }
            })

        return documents
