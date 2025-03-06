import os
import pickle
from docling.document_converter import DocumentConverter
from langchain_experimental.text_splitter import SemanticChunker

def convert_pdf_to_markdown(file_path):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

def process_and_chunk_document(markdown_content, embeddings, cache_filename):
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type='percentile')
    try:
        with open(cache_filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        docs = text_splitter.create_documents([markdown_content])
        with open(cache_filename, "wb") as f:
            pickle.dump(docs, f)
        return docs