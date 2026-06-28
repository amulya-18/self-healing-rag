"""
ingest.py
Reads documents from the docs/ folder, splits them into chunks,
embeds them locally, and saves a FAISS index to disk.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DOCS_FOLDER = "docs"
INDEX_FOLDER = "faiss_index"


def load_documents():
    """Load every PDF and TXT file from the docs/ folder."""
    documents = []
    for filename in os.listdir(DOCS_FOLDER):
        filepath = os.path.join(DOCS_FOLDER, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
            print(f"Loaded PDF: {filename}")

        elif filename.lower().endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
            print(f"Loaded TXT: {filename}")

    return documents


def split_documents(documents):
    """Break documents into small overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # roughly 500 characters per chunk
        chunk_overlap=50,    # slight overlap so context isn't cut off mid-idea
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_and_save_index(chunks):
    """Embed chunks locally and save a FAISS index to disk."""
    print("Loading local embedding model (first run downloads it, ~80MB)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding chunks and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(INDEX_FOLDER)
    print(f"Index saved to ./{INDEX_FOLDER}")


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("No documents found in docs/ folder. Add a PDF or TXT file and try again.")
    else:
        chunks = split_documents(docs)
        build_and_save_index(chunks)
        print("Ingestion complete!")
