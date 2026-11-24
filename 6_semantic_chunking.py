"""
ÉTAPE 6 : CHUNKING SÉMANTIQUE (SemanticChunker)
Exécution : python 6_semantic_chunking.py
"""

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from display import show_header, show_text_chunks

load_dotenv()

# Modèle d'embeddings Google Gemini (Free Tier)
EMBEDDING_MODEL = "gemini-embedding-001"

SAMPLE_TEXT = """Produit matriciel et propriétés.
Le produit AB n'est défini que si le nombre de colonnes de A est égal au nombre de lignes de B.
En général, le produit matriciel n'est pas commutatif : AB est différent de BA.
En revanche, il est associatif et la trace vérifie toujours tr(AB) = tr(BA).

Déterminant et matrice inversible.
Pour une matrice 2x2 [[a, b], [c, d]], le déterminant vaut ad - bc.
Une matrice carrée est inversible si et seulement si son déterminant est non nul.
Géométriquement, la valeur absolue du déterminant représente l'aire ou le volume de la figure transformée.

Valeurs propres et diagonalisation.
Un scalaire lambda est valeur propre de A s'il existe un vecteur non nul v tel que Av = lambda v.
La diagonalisation écrit A = P D P^(-1), ce qui permet de calculer très facilement les puissances A^n.
Ces notions sont au cœur de l'ACP (PCA) en data science et de l'algorithme PageRank."""


def main():
    show_header("ÉTAPE 6 : CHUNKING SÉMANTIQUE (EMBEDDINGS GEMINI)")

    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    semantic_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",  # ou "standard_deviation"
        breakpoint_threshold_amount=70,
    )

    chunks = semantic_splitter.split_text(SAMPLE_TEXT)
    show_text_chunks("Résultats du Chunking Sémantique", chunks)


if __name__ == "__main__":
    main()
