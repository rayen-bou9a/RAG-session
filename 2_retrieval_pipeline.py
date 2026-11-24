"""
ÉTAPE 2 : PIPELINE DE RECHERCHE VECTORIELLE (Retrieval)
-------------------------------------------------------
Objectif :
    Interroger la base vectorielle ChromaDB pour retrouver les passages du cours
    les plus proches sémantiquement d'une question d'étudiant.

Exécution :
    python 2_retrieval_pipeline.py
"""

from langchain_chroma import Chroma

from config import DB_DIR, get_embeddings, print_header


def main():
    print_header("ÉTAPE 2 : RECHERCHE VECTORIELLE (RETRIEVAL)")

    # 1. Connexion à la base vectorielle existante
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 2. Question posée par l'étudiant
    query = "Quelles sont les valeurs propres de la matrice Q et comment calculer Q^5 ?"

    # 3. Configuration du retriever (top-k chunks les plus similaires)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Variante avec seuil de similarité minimal :
    # retriever = db.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={"k": 3, "score_threshold": 0.3},
    # )

    relevant_docs = retriever.invoke(query)

    # 4. Affichage des résultats retrouvés
    print(f"\nQuestion : {query}")
    print(f"Extraits retrouvés : {len(relevant_docs)}\n")

    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get("source", "inconnu")
        print(f"--- Extrait {i} (Source : {source}) ---")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()


# =============================================================================
# Questions d'entraînement (issues des 4 chapitres du cours sur les matrices) :
# =============================================================================
# 1. "Pourquoi le produit matriciel AB est-il différent de BA ? Donne l'exemple du cours."
# 2. "Quelle est la complexité de l'algorithme de Strassen pour le produit matriciel ?"
# 3. "Quelle est l'interprétation géométrique du déterminant d'une matrice 2x2 et 3x3 ?"
# 4. "Pourquoi la matrice M = [[2, 0, 1], [1, 3, 2], [1, 1, 1]] n'est-elle pas inversible ?"
# 5. "Quelle est la solution unique du système S par la méthode du pivot de Gauss ?"
# 6. "Que dit le théorème de Rouché-Capelli sur l'existence et l'unicité des solutions ?"
# 7. "Quelles sont les valeurs propres de la matrice Q = [[2, 1], [1, 2]] et comment calculer Q^5 ?"
# 8. "Quel est le lien entre les valeurs propres, l'ACP (PCA), PageRank et le RAG ?"
