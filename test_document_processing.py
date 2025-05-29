# test_document_processing.py - Test your PDF processing

import os
from src.document_processor import DocumentProcessor


def test_pdf_processing():
    """Test PDF processing with a sample document"""
    processor = DocumentProcessor()

    # Check if you have any PDFs in data/raw/
    pdf_dir = "data/raw"
    if not os.path.exists(pdf_dir):
        print(f"❌ Directory {pdf_dir} doesn't exist. Create it and add some PDFs!")
        return

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_dir}")
        print("Download a research paper PDF and put it in data/raw/")
        return

    # Process the first PDF
    pdf_path = os.path.join(pdf_dir, pdf_files[0])
    print(f"🔍 Processing: {pdf_files[0]}")

    try:
        # Extract text
        text = processor.extract_text_from_pdf(pdf_path)
        print(f"✅ Extracted {len(text)} characters")
        print(f"📝 First 200 chars: {text[:200]}...")

        # Create chunks
        documents = processor.process_pdf(pdf_path)
        print(f"✅ Created {len(documents)} chunks")

        # Show first chunk
        if documents:
            first_chunk = documents[0]
            print("📄 First chunk preview:")
            print(f"   Source: {first_chunk['metadata']['source']}")
            print(f"   Content: {first_chunk['content'][:200]}...")

        return True

    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return False


if __name__ == "__main__":
    test_pdf_processing()
