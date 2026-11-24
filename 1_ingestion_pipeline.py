"""
ÉTAPE 1 : PIPELINE D'INGESTION (Document Loading -> Chunking -> Vector Store)
-----------------------------------------------------------------------------
Objectif :
    1. Charger les chapitres du cours d'algèbre linéaire depuis le dossier `docs/`.
    2. Découper chaque document en morceaux (chunks).
    3. Calculer les embeddings avec Gemini et les stocker dans ChromaDB.

Exécution :
    python 1_ingestion_pipeline.py
"""

import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter

from config import DB_DIR, DOCS_DIR, get_embeddings, print_header


def load_documents(docs_path: str = DOCS_DIR):
    """Charge tous les fichiers .txt du dossier `docs/`."""
    print(f"Chargement des documents depuis '{docs_path}/'...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Le dossier '{docs_path}' n'existe pas. Ajoutez vos fichiers .txt dedans."
        )

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    if not documents:
        raise FileNotFoundError(f"Aucun fichier .txt trouvé dans '{docs_path}/'.")

    print(f"-> {len(documents)} documents chargés :")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc.metadata['source']} ({len(doc.page_content)} caractères)")

    return documents


def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 100):
    """Découpe les documents en chunks avec un léger chevauchement (overlap)."""
    print(f"\nDécoupage en chunks (taille={chunk_size}, overlap={chunk_overlap})...")

    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    print(f"-> {len(chunks)} chunks créés. Aperçu des 3 premiers :")
    for i, chunk in enumerate(chunks[:3], 1):
        preview = chunk.page_content.replace("\n", " ")[:120]
        print(f"\n  [Chunk {i}] Source : {chunk.metadata['source']} ({len(chunk.page_content)} chars)")
        print(f"  Aperçu : {preview}...")

    return chunks


def create_vector_store(chunks, persist_directory: str = DB_DIR):
    """Génère les embeddings et sauvegarde la base vectorielle ChromaDB."""
    print("\nCalcul des embeddings (Gemini) et stockage dans ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"-> Base vectorielle sauvegardée dans '{persist_directory}'.")
    return vectorstore


def main():
    print_header("ÉTAPE 1 : INGESTION DES DOCUMENTS (COURS MATRICES)")

    if os.path.exists(DB_DIR):
        print(f"La base '{DB_DIR}' existe déjà. Chargement sans ré-indexation...")
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=get_embeddings(),
            collection_metadata={"hnsw:space": "cosine"},
        )
        print(f"-> {vectorstore._collection.count()} chunks disponibles dans ChromaDB.")
        print("   (Supprimez le dossier 'db/' si vous modifiez les fichiers de 'docs/'.)")
        return vectorstore

    # 1. Charger les chapitres du cours
    documents = load_documents(DOCS_DIR)

    # 2. Découper en chunks
    chunks = split_documents(documents)

    # 3. Indexer dans ChromaDB
    vectorstore = create_vector_store(chunks, DB_DIR)

    print("\nIngestion terminée ! Vos documents sont prêts pour le RAG.")
    return vectorstore


if __name__ == "__main__":
    main()
