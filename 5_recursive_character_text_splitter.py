"""
ÉTAPE 5 : DÉCOUPAGE RÉCURSIF (RecursiveCharacterTextSplitter)
-------------------------------------------------------------
Objectif :
    Comparer le découpage simple (`CharacterTextSplitter`) avec le découpage
    récursif (`RecursiveCharacterTextSplitter`), qui essaie plusieurs séparateurs
    dans l'ordre (paragraphes, lignes, phrases, mots) pour respecter la taille maximale.

Exécution :
    python 5_recursive_character_text_splitter.py
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from config import print_header

SAMPLE_TEXT = """Produit matriciel

Le produit AB n'est défini que si le nombre de colonnes de A est égal au nombre de lignes de B.

Déterminant et inverse

Une matrice carrée A est inversible si et seulement si son déterminant det(A) est différent de 0.

Pivot de Gauss

La méthode du pivot de Gauss transforme le système AX = b en une matrice triangulaire supérieure grâce à des opérations élémentaires sur les lignes sans jamais modifier l'ensemble des solutions du système linéaire étudié."""


def show_chunks(chunks):
    for i, chunk in enumerate(chunks, 1):
        print(f'Chunk {i} ({len(chunk)} caractères) :\n"{chunk}"\n')


def main():
    print_header("ÉTAPE 5 : CHARACTER vs RECURSIVE CHARACTER TEXT SPLITTER")

    # 1. Découpage simple (échoue à couper le dernier paragraphe trop long)
    print("\n1) CharacterTextSplitter (un seul séparateur '\\n\\n') :\n")
    simple_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=120,
        chunk_overlap=0,
    )
    show_chunks(simple_splitter.split_text(SAMPLE_TEXT))

    # 2. Découpage récursif (essaie \n\n, puis \n, puis '. ', puis ' ')
    print("-" * 65)
    print("2) RecursiveCharacterTextSplitter (plusieurs séparateurs) :\n")
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=120,
        chunk_overlap=0,
    )
    show_chunks(recursive_splitter.split_text(SAMPLE_TEXT))


if __name__ == "__main__":
    main()
