"""
ÉTAPE 3 : GÉNÉRATION AUGMENTÉE PAR LA RECHERCHE (RAG complet)
-------------------------------------------------------------
Objectif :
    Combiner la recherche vectorielle (ChromaDB) et le LLM (Gemini) pour
    répondre à une question en s'appuyant UNIQUEMENT sur les extraits du cours.

Exécution :
    python 3_answer_generation.py
"""

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from config import DB_DIR, extract_text, get_embeddings, get_llm, print_header


def main():
    print_header("ÉTAPE 3 : GÉNÉRATION DE RÉPONSE (RAG)")

    # 1. Charger la base vectorielle
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 2. Rechercher les extraits pertinents
    query = "Quelles sont les valeurs propres de la matrice Q et comment calculer Q^5 ?"
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)

    print(f"\nQuestion : {query}\n")
    print("--- Contexte retrouvé ---")
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get("source", "inconnu")
        print(f"\n[Extrait {i} — {source}]")
        print(doc.page_content)

    # 3. Construire le prompt augmenté (Question + Extraits du cours)
    context_text = "\n\n".join(f"- {doc.page_content}" for doc in relevant_docs)
    user_prompt = f"""En te basant uniquement sur les extraits de cours suivants, réponds clairement à la question : {query}

Extraits du cours :
{context_text}

Consignes :
- Utilise uniquement les informations présentes dans les extraits ci-dessus.
- Si l'information ne s'y trouve pas, réponds : "Je n'ai pas assez d'informations dans les documents fournis pour répondre à cette question."
"""

    # 4. Appeler le modèle Gemini
    model = get_llm()
    messages = [
        SystemMessage(content="Tu es un assistant pédagogique spécialisé en algèbre linéaire."),
        HumanMessage(content=user_prompt),
    ]
    result = model.invoke(messages)

    # 5. Afficher la réponse proprement
    print("\n--- Réponse générée par Gemini ---")
    print(extract_text(result.content))


if __name__ == "__main__":
    main()
