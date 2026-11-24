"""
ÉTAPE 5 : DÉCOUPAGE RÉCURSIF (RecursiveCharacterTextSplitter)
Exécution : python 5_recursive_character_text_splitter.py
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from display import show_header, show_text_chunks

SAMPLE_TEXT = """Produit matriciel

Le produit AB n'est défini que si le nombre de colonnes de A est égal au nombre de lignes de B.

Déterminant et inverse

Une matrice carrée A est inversible si et seulement si son déterminant det(A) est différent de 0.

Pivot de Gauss

La méthode du pivot de Gauss transforme le système AX = b en une matrice triangulaire supérieure grâce à des opérations élémentaires sur les lignes sans jamais modifier l'ensemble des solutions du système linéaire étudié."""


def main():
    show_header("ÉTAPE 5 : CHARACTER vs RECURSIVE CHARACTER TEXT SPLITTER")

    # 1. Découpage simple (un seul séparateur)
    simple_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=120,
        chunk_overlap=0,
    )
    show_text_chunks("CharacterTextSplitter", simple_splitter.split_text(SAMPLE_TEXT))

    # 2. Découpage récursif (essaie plusieurs séparateurs dans l'ordre)
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=120,
        chunk_overlap=0,
    )
    show_text_chunks("RecursiveCharacterTextSplitter", recursive_splitter.split_text(SAMPLE_TEXT))


if __name__ == "__main__":
    main()
