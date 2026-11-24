"""
ÉTAPE 3 : GÉNÉRATION AUGMENTÉE PAR LA RECHERCHE (RAG complet)
Exécution : python 3_answer_generation.py
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from display import extract_text, show_answer, show_header, show_retrieved_docs

load_dotenv()

# Configuration des modèles Google Gemini (Free Tier)
DB_DIR = "db/chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"


def main():
    show_header("ÉTAPE 3 : GÉNÉRATION DE RÉPONSE (RAG)")

    # 1. Charger la base vectorielle
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 2. Retrouver les extraits pertinents
    query = "Quelles sont les valeurs propres de la matrice Q et comment calculer Q^5 ?"
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)
    show_retrieved_docs(query, relevant_docs)

    # 3. Construire le prompt augmenté
    context_text = "\n\n".join(f"- {doc.page_content}" for doc in relevant_docs)
    user_prompt = f"""En te basant uniquement sur les extraits de cours suivants, réponds clairement à la question : {query}

                    Extraits du cours :
                    {context_text}

                    Consignes :
                    - Utilise uniquement les informations présentes dans les extraits ci-dessus.
                    - Si l'information ne s'y trouve pas, réponds : "Je n'ai pas assez d'informations dans les documents fournis pour répondre à cette question."
                    """

    # 4. Appeler Gemini pour générer la réponse
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL)
    messages = [
        SystemMessage(content="Tu es un assistant pédagogique spécialisé en algèbre linéaire."),
        HumanMessage(content=user_prompt),
    ]
    result = model.invoke(messages)

    # 5. Afficher la réponse
    show_answer(extract_text(result.content))


if __name__ == "__main__":
    main()
