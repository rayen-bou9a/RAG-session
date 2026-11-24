"""
ÉTAPE 4 : RAG CONVERSATIONNEL AVEC MÉMOIRE (History-Aware Generation)
Exécution : python 4_history_aware_generation.py
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from display import extract_text, show_chat_turn, show_header

load_dotenv()

# Configuration des modèles Google Gemini (Free Tier)
DB_DIR = "db/chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"


def build_standalone_question(model, chat_history, user_question: str) -> str:
    """Reformule la question à partir de l'historique pour améliorer la recherche."""
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
    return extract_text(model.invoke(messages).content).strip()


def ask_question(db, model, chat_history, user_question: str) -> str:
    """Exécute un tour complet de RAG conversationnel."""
    # 1. Reformulation de la question si un historique existe
    search_question = build_standalone_question(model, chat_history, user_question)

    # 2. Recherche vectorielle
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    # 3. Construction du prompt et génération de la réponse
    context_text = "\n\n".join(f"- {doc.page_content}" for doc in docs)
    prompt = f"""En te basant uniquement sur les extraits de cours suivants, réponds à la question : {user_question}

Extraits du cours :
{context_text}

Si l'information ne se trouve pas dans les extraits, dis simplement que tu n'as pas assez d'informations dans les documents fournis.
"""

    messages = [
        SystemMessage(
            content=(
                "Tu es un assistant pédagogique en algèbre linéaire. "
                "Réponds en t'appuyant sur les extraits du cours et l'historique."
            )
        ),
        *chat_history,
        HumanMessage(content=prompt),
    ]
    answer = extract_text(model.invoke(messages).content)

    # 4. Mémorisation de l'échange et affichage
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    show_chat_turn(user_question, search_question, docs, answer)
    return answer


def main():
    show_header("ÉTAPE 4 : CHATBOT RAG SUR LE COURS DE MATRICES")
    print("Posez vos questions sur les matrices ! Tapez 'quit' pour quitter.")

    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL)
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
