"""
ÉTAPE 1 : PIPELINE D'INGESTION (Document Loading -> Chunking -> Vector Store)
Exécution : python 1_ingestion_pipeline.py
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from display import (
    show_chunks_preview,
    show_existing_db,
    show_header,
    show_ingestion_done,
    show_loaded_documents,
)

load_dotenv()

# Configuration du pipeline
DOCS_DIR = "docs"
DB_DIR = "db/chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"


def load_documents(docs_path: str = DOCS_DIR):
    """Charge tous les fichiers .txt du dossier `docs/`."""
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Le dossier '{docs_path}' n'existe pas.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    if not documents:
        raise FileNotFoundError(f"Aucun fichier .txt trouvé dans '{docs_path}/'.")

    show_loaded_documents(documents)
    return documents


def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 100):
    """Découpe les documents en chunks avec chevauchement (overlap)."""
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    show_chunks_preview(chunks)
    return chunks


def create_vector_store(chunks, persist_directory: str = DB_DIR):
    """Génère les embeddings Gemini et sauvegarde dans ChromaDB."""
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    show_ingestion_done(persist_directory)
    return vectorstore


def main():
    show_header("ÉTAPE 1 : INGESTION DES DOCUMENTS (COURS MATRICES)")

    if os.path.exists(DB_DIR):
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
        show_existing_db(DB_DIR, vectorstore._collection.count())
        return vectorstore

    # 1. Charger les chapitres du cours
    documents = load_documents(DOCS_DIR)

    # 2. Découper en chunks
    chunks = split_documents(documents)

    # 3. Créer les embeddings et stocker dans ChromaDB
    return create_vector_store(chunks, DB_DIR)


if __name__ == "__main__":
    main()
