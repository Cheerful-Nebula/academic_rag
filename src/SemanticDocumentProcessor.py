from .SemanticChunker import SemanticChunker
import os
import pdfplumber
import re
import concurrent.futures
"""
This module provides a document processor that extracts text from PDF files
and processes it into semantic chunks using the SemanticChunker class.
"""


class SemanticDocumentProcessor:
    def __init__(self):
        """Initialize the document processor with semantic chunking."""
        self.semantic_chunker = SemanticChunker()
        # self.logger = logging.getLogger('semantic_processor')

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file using pdfplumber."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # !test affect of cropping, effective_box, on page text extract,page_text!
                    effective_box = page.cropbox or page.mediabox
                    cropped_page = page.crop(effective_box)
                    page_text = cropped_page.extract_text(x_tolerance=2, use_text_flow=True)
                    if page_text:
                        text += page_text + "\n"
            return self.clean_text(text.strip())
        except Exception as e:
            raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")

    def clean_text(self, text: str, preserve_structure: bool = False) -> str:
        text = re.sub(r'arXiv:\d{4}\.\d{4,5}v\d+\s+\d+\s+\S+\s+\d+', '', text)
        text = re.sub(r'\S+@\S+', '', text)

        if preserve_structure:
            # Gentle cleaning for formula-rich papers
            return re.sub(r'[ \t]+', ' ', text).strip()
        else:
            # Aggressive cleaning for text-heavy papers
            return re.sub(r'\s+', ' ', text).strip()

    def _should_preserve_structure(self, text: str) -> bool:
        """
        Determine if the PDF's structure should be preserved based on content analysis.
        Returns True if the document appears to be formula-heavy or contains special formatting.
        """
        # Check for mathematical symbols and formulas
        math_indicators = {
            r'\$.*?\$',           # LaTeX inline math
            r'\\\[.*?\\\]',       # LaTeX display math
            r'[∑∏∫∂√α-ωΑ-Ω]',    # Mathematical symbols
            r'\d+(?:\.\d+)?[×÷±∓∝∞≈≠≤≥]',  # Numbers with math operators
        }

        # Check for technical patterns
        technical_indicators = {
            r'(?:Algorithm|Theorem|Lemma|Proof)\s+\d+',  # Formal statements
            r'\\begin\{.*?\}',    # LaTeX environments
            r'Fig\.\s+\d+',       # Figure references
            r'Table\s+\d+',       # Table references
            r'Equation\s+\d+',    # Equation references
        }

        # Calculate the ratio of special characters to text length
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text)

        # Check for matches in our indicator patterns
        has_math = any(bool(re.search(pattern, text)) for pattern in math_indicators)
        has_technical = any(bool(re.search(pattern, text)) for pattern in technical_indicators)

        # Decision criteria
        return (
            has_math or
            has_technical or
            special_char_ratio > 0.1  # More than 10% special characters
        )

    def process_pdf(self, pdf_path: str, force_preserve_structure: bool = None) -> list[dict]:
        """
        Complete PDF processing pipeline with semantic chunking.

        Args:
            pdf_path (str): Path to the PDF file
            force_preserve_structure (bool, optional): Force structure preservation mode.
                If None, automatically determines based on content analysis.
        """
        # Path of PDF that will be processed
        filename = os.path.basename(pdf_path)
        print(f"Processing {filename} with semantic chunking...")

        # First pass to get raw text for structure analysis
        try:
            with pdfplumber.open(pdf_path) as pdf:
                raw_text = "\n".join(
                    page.extract_text(x_tolerance=2, use_text_flow=True) or ""
                    for page in pdf.pages
                )
        except Exception as e:
            print(f"Error in initial text extraction from {filename}: {str(e)}")
            return []

        # Determine if we should preserve structure
        should_preserve = (
            force_preserve_structure if force_preserve_structure is not None
            else self._should_preserve_structure(raw_text)
        )

        print(f"Structure preservation {'enabled' if should_preserve else 'disabled'} "
              f"for {filename}")

        # Extract and clean text with determined structure preservation
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"No text extracted from {filename}.")
            return []

        # Clean text with determined structure preservation
        text = self.clean_text(text.strip(), preserve_structure=should_preserve)

        # Use semantic chunking
        documents = self.semantic_chunker.chunk_text_semantically(text, filename)

        # Add structure preservation info to metadata
        for doc in documents:
            doc['metadata']['structure_preserved'] = should_preserve

        print(f"Processed {filename} into {len(documents)} semantic chunks.")
        print(f"Chunk types: {set(doc['metadata']['chunk_type'] for doc in documents)}")
        print(f"Chunk sections: {set(doc['metadata']['section'] for doc in documents)}")

        return documents

        # Use semantic chunking instead of fixed-size chunking
        documents = self.semantic_chunker.chunk_text_semantically(text, filename)

        print(f"Processed {filename} into {len(documents)} semantic chunks.")
        print(f"Chunk types: {set(doc['metadata']['chunk_type'] for doc in documents)}")
        print(f"Chunk sections: {set(doc['metadata']['section'] for doc in documents)}")

        return documents

    def batch_process(
        self,
        pdf_paths: list[str],
        max_workers: int = 4,
        force_preserve_structure: bool = None
    ) -> list[dict]:
        """
        Process a batch of PDF files concurrently using ThreadPoolExecutor.

        Args:
            pdf_paths (list[str]): List of PDF file paths to process.
            max_workers (int): Maximum number of worker threads to use.
            force_preserve_structure (bool, optional): Force structure preservation mode.
                If None, automatically determines based on content analysis.

        Returns:
            list[dict]: List of processed documents with semantic chunks.
        """
        max_workers = max_workers or min(4, len(pdf_paths))
        print(f"Batch processing {len(pdf_paths)} PDFs with up to {max_workers} workers...")

        if not pdf_paths:
            print("No PDF files to process.")
            return []

        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if len(pdf_paths) == 1:
            print("Only one PDF to process, using single-threaded execution.")
            return self.process_pdf(
                pdf_paths[0],
                force_preserve_structure=force_preserve_structure
            )

        # Use ThreadPoolExecutor to process PDFs concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a processing function with the force_preserve_structure parameter
            def process_with_params(path):
                return self.process_pdf(
                    path,
                    force_preserve_structure=force_preserve_structure
                )
            results = list(executor.map(process_with_params, pdf_paths))

        return [doc for sublist in results for doc in sublist]
