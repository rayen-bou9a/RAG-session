"""
ÉTAPE 2 : PIPELINE DE RECHERCHE VECTORIELLE (Retrieval)
Exécution : python 2_retrieval_pipeline.py
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from display import show_header, show_retrieved_docs

load_dotenv()

# Configuration
DB_DIR = "db/chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"


def main():
    show_header("ÉTAPE 2 : RECHERCHE VECTORIELLE (RETRIEVAL)")

    # 1. Connexion à la base vectorielle ChromaDB
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 2. Question et recherche des k extraits les plus proches
    query = "Pourquoi le produit matriciel AB est-il différent de BA ? Donne l'exemple du cours."
    retriever = db.as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(query)

    # 3. Affichage
    show_retrieved_docs(query, relevant_docs)


if __name__ == "__main__":
    main()


# Questions d'entraînement sur les 4 chapitres du cours :
# 1. "Pourquoi le produit matriciel AB est-il différent de BA ? Donne l'exemple du cours."
# 2. "Quelle est la complexité de l'algorithme de Strassen pour le produit matriciel ?"
# 3. "Quelle est l'interprétation géométrique du déterminant d'une matrice 2x2 et 3x3 ?"
# 4. "Pourquoi la matrice M = [[2, 0, 1], [1, 3, 2], [1, 1, 1]] n'est-elle pas inversible ?"
# 5. "Quelle est la solution unique du système S par la méthode du pivot de Gauss ?"
# 6. "Que dit le théorème de Rouché-Capelli sur l'existence et l'unicité des solutions ?"
# 7. "Quelles sont les valeurs propres de la matrice Q = [[2, 1], [1, 2]] et comment calculer Q^5 ?"
# 8. "Quel est le lien entre les valeurs propres, l'ACP (PCA), PageRank et le RAG ?"
