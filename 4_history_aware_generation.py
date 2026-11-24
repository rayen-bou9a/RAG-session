"""
ÉTAPE 4 : RAG CONVERSATIONNEL AVEC MÉMOIRE (History-Aware Generation)
---------------------------------------------------------------------
Objectif :
    Discuter avec le cours d'algèbre linéaire en gardant l'historique :
    - Reformule les questions de suivi ("Et ses vecteurs propres ?") en question autonome.
    - Recherche les bons extraits dans ChromaDB avec la question reformulée.
    - Génère la réponse avec Gemini en tenant compte de la conversation.

Exécution :
    python 4_history_aware_generation.py
"""

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import DB_DIR, extract_text, get_embeddings, get_llm, print_header


def build_standalone_question(model, chat_history, user_question: str) -> str:
    """Reformule la question en tenant compte de l'historique pour améliorer la recherche."""
    if not chat_history:
        return user_question

    messages = [
        SystemMessage(
            content=(
                "À partir de l'historique de conversation, reformule la nouvelle question "
                "pour qu'elle soit autonome et claire pour une recherche documentaire. "
                "Retourne uniquement la question reformulée."
            )
        ),
        *chat_history,
        HumanMessage(content=f"Nouvelle question : {user_question}"),
    ]
    result = model.invoke(messages)
    return extract_text(result.content).strip()


def ask_question(db, model, chat_history, user_question: str) -> str:
    """Exécute un tour complet de RAG conversationnel."""
    print(f"\n--- Question : {user_question} ---")

    # 1. Reformulation si un historique existe
    search_question = build_standalone_question(model, chat_history, user_question)
    if search_question != user_question:
        print(f"Recherche reformulée : {search_question}")

    # 2. Recherche vectorielle
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Extraits trouvés : {len(docs)}")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "inconnu")
        preview = doc.page_content.replace("\n", " ")[:90]
        print(f"  {i}. [{source}] {preview}...")

    # 3. Construction du prompt final
    context_text = "\n\n".join(f"- {doc.page_content}" for doc in docs)
    prompt = f"""En te basant uniquement sur les extraits de cours suivants, réponds à la question : {user_question}

Extraits du cours :
{context_text}

Si l'information ne se trouve pas dans les extraits, dis simplement que tu n'as pas assez d'informations dans les documents fournis.
"""

    # 4. Génération de la réponse
    messages = [
        SystemMessage(
            content=(
                "Tu es un assistant pédagogique en algèbre linéaire. "
                "Réponds en t'appuyant sur les extraits du cours et l'historique de conversation."
            )
        ),
        *chat_history,
        HumanMessage(content=prompt),
    ]
    result = model.invoke(messages)
    answer = extract_text(result.content)

    # 5. Mise à jour de l'historique
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"\nRéponse :\n{answer}")
    return answer


def main():
    print_header("ÉTAPE 4 : CHATBOT RAG SUR LE COURS DE MATRICES")
    print("Posez vos questions sur les matrices ! Tapez 'quit' pour quitter.")
    print("Exemple de suivi : demandez d'abord 'Quelles sont les valeurs propres de Q ?'")
    print("puis demandez 'Et quels sont ses vecteurs propres ?'\n")

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    model = get_llm()
    chat_history = []

    while True:
        question = input("\nVotre question : ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Au revoir !")
            break

        ask_question(db, model, chat_history, question)


if __name__ == "__main__":
    main()
