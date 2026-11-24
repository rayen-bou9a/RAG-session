"""
ÉTAPE 7 : CHUNKING AGENTIQUE (piloté par un LLM)
Exécution : python 7_agentic_chunking.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from display import extract_text, show_header, show_text_chunks

load_dotenv()

# Modèle de chat Google Gemini (Free Tier)
CHAT_MODEL = "gemini-3.5-flash-lite"

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
    show_header("ÉTAPE 7 : CHUNKING AGENTIQUE (PILOTÉ PAR GEMINI)")

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)

    prompt = f"""Tu es un expert en découpage de texte pour un système RAG.
Découpe le texte ci-dessous en chunks logiques et cohérents.

Règles :
- Regroupe ensemble les informations qui traitent du même thème mathématique.
- Chaque chunk doit rester concis (environ 250 à 350 caractères maximum).
- Insère exactement le marqueur <<<SPLIT>>> entre chaque chunk.
- Ne modifie pas le contenu original et n'ajoute aucun commentaire.

Texte :
{SAMPLE_TEXT}
"""

    response = llm.invoke(prompt)
    marked_text = extract_text(response.content)
    chunks = [chunk.strip() for chunk in marked_text.split("<<<SPLIT>>>") if chunk.strip()]

    show_text_chunks("Résultats du Chunking Agentique", chunks)


if __name__ == "__main__":
    main()
