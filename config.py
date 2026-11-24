"""Configuration partagée pour le Workshop RAG (Algèbre Linéaire — Matrices)."""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Charge les variables d'environnement depuis le fichier .env (GOOGLE_API_KEY)
load_dotenv()

# Chemins du projet
DOCS_DIR = "docs"
DB_DIR = "db/chroma_db"

# Modèles Google Gemini (Free Tier)
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Retourne le modèle d'embeddings Gemini."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm() -> ChatGoogleGenerativeAI:
    """Retourne le modèle de chat Gemini."""
    return ChatGoogleGenerativeAI(model=CHAT_MODEL)


def extract_text(content) -> str:
    """Extrait proprement le texte brut d'une réponse Gemini (str ou liste de blocs)."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict)
    )


def print_header(title: str) -> None:
    """Affiche un bandeau lisible dans le terminal."""
    line = "=" * 65
    print(f"\n{line}\n  {title}\n{line}")
