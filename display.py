"""Fonctions d'affichage dans le terminal pour garder les scripts du workshop lisibles."""


def extract_text(content) -> str:
    """Extrait proprement le texte brut d'une réponse Gemini (str ou liste de blocs)."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict)
    )


def show_header(title: str) -> None:
    """Affiche un bandeau de titre."""
    line = "=" * 65
    print(f"\n{line}\n  {title}\n{line}")


def show_loaded_documents(documents) -> None:
    """Affiche le résumé des documents chargés."""
    print(f"\n-> {len(documents)} documents chargés :")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc.metadata['source']} ({len(doc.page_content)} caractères)")


def show_chunks_preview(chunks, max_items: int = 3) -> None:
    """Affiche un aperçu des premiers chunks créés lors de l'ingestion."""
    print(f"\n-> {len(chunks)} chunks créés (aperçu des {min(max_items, len(chunks))} premiers) :")
    for i, chunk in enumerate(chunks[:max_items], 1):
        preview = chunk.page_content.replace("\n", " ")[:120]
        print(f"\n  [Chunk {i}] {chunk.metadata['source']} ({len(chunk.page_content)} chars)")
        print(f"  {preview}...")


def show_existing_db(db_dir: str, count: int) -> None:
    """Indique que la base ChromaDB existe déjà."""
    print(f"\nLa base '{db_dir}' existe déjà ({count} chunks disponibles).")
    print("Supprimez le dossier 'db/' si vous modifiez les fichiers de 'docs/'.")


def show_ingestion_done(db_dir: str) -> None:
    """Confirme la fin de l'ingestion."""
    print(f"\n-> Base vectorielle sauvegardée dans '{db_dir}'.")
    print("Ingestion terminée ! Vos documents sont prêts pour le RAG.")


def show_retrieved_docs(query: str, docs) -> None:
    """Affiche la question et les extraits retrouvés par ChromaDB."""
    print(f"\nQuestion : {query}")
    print(f"Extraits retrouvés : {len(docs)}\n")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "inconnu")
        print(f"--- Extrait {i} ({source}) ---")
        print(doc.page_content)
        print()


def show_answer(answer: str) -> None:
    """Affiche la réponse générée par Gemini."""
    print("--- Réponse générée par Gemini ---")
    print(answer)


def show_chat_turn(user_question: str, search_question: str, docs, answer: str) -> None:
    """Affiche un échange complet du chatbot RAG."""
    print(f"\n--- Question : {user_question} ---")
    if search_question != user_question:
        print(f"Recherche reformulée : {search_question}")

    print(f"Extraits utilisés ({len(docs)}) :")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "inconnu")
        preview = doc.page_content.replace("\n", " ")[:90]
        print(f"  {i}. [{source}] {preview}...")

    print(f"\nRéponse :\n{answer}")


def show_text_chunks(title: str, chunks) -> None:
    """Affiche une liste de chunks texte (scripts 5, 6 et 7)."""
    print(f"\n--- {title} ({len(chunks)} chunks) ---\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"[Chunk {i} — {len(chunk)} caractères]")
        print(chunk.strip())
        print()
