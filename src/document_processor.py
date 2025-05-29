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
