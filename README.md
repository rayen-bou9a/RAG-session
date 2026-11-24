# Workshop RAG (Retrieval-Augmented Generation) avec Google Gemini

Ce dépôt contient les scripts pratiques pour le workshop sur le **RAG (Retrieval-Augmented Generation)** avec **LangChain**, **ChromaDB** et les **modèles gratuits de Google Gemini**.

---

## Prérequis et installation

> **Pour les étudiants :** suivez le guide détaillé pas à pas dans **[`SETUP.md`](SETUP.md)** pour créer votre environnement virtuel (`venv`), installer les dépendances et configurer votre clé API Gemini.

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Configurer la clé API Gemini (gratuit)

1. Créez une clé API gratuite sur [Google AI Studio](https://aistudio.google.com/apikey).
2. Copiez le fichier `.env.example` en `.env` :
   ```bash
   cp .env.example .env
   ```
3. Renseignez votre clé dans `.env` :
   ```env
   GOOGLE_API_KEY=votre_cle_api_ici
   ```

> **Sécurité :** Le fichier `.env` est listé dans `.gitignore` et ne doit **jamais** être commité sur GitHub.

### Modèles Gemini utilisés (Free Tier)

| Usage | Modèle Gemini | Classe LangChain |
|---|---|---|
| **Génération (LLM)** | `gemini-3.5-flash-lite` | `ChatGoogleGenerativeAI` |
| **Embeddings** | `gemini-embedding-001` | `GoogleGenerativeAIEmbeddings` |

---

## Structure du Workshop (les 7 notions)

Chaque script illustre une étape ou une stratégie clé du RAG, dans l'ordre progressif :

### Partie 1 — Pipeline RAG de base

1. **[`1_ingestion_pipeline.py`](1_ingestion_pipeline.py)** — *Ingestion des documents*
   - Charge les fichiers `.txt` depuis le dossier `docs/`.
   - Découpe les documents en morceaux (*chunks*) avec `CharacterTextSplitter`.
   - Génère les vecteurs (*embeddings*) via `gemini-embedding-001` et les stocke localement dans **ChromaDB** (`db/chroma_db`).

2. **[`2_retrieval_pipeline.py`](2_retrieval_pipeline.py)** — *Recherche vectorielle (Retrieval)*
   - Recharge la base vectorielle ChromaDB.
   - Transforme la question de l'utilisateur en vecteur et récupère les `k` chunks les plus pertinents par similarité cosinus.

3. **[`3_answer_generation.py`](3_answer_generation.py)** — *Génération augmentée (RAG complet)*
   - Combine la question et les documents retrouvés dans un prompt structuré.
   - Envoie le contexte à `gemini-3.5-flash-lite` pour générer une réponse fidèle aux documents fournis.

4. **[`4_history_aware_generation.py`](4_history_aware_generation.py)** — *RAG conversationnel avec mémoire*
   - Boucle de chat interactive conservant l'historique (`chat_history`).
   - Reformule automatiquement les questions de suivi en questions autonomes (*standalone*) avant la recherche vectorielle.

### Partie 2 — Stratégies avancées de Chunking

5. **[`5_recursive_character_text_splitter.py`](5_recursive_character_text_splitter.py)** — *Découpage récursif*
   - Compare le découpage simple avec `RecursiveCharacterTextSplitter`, qui essaie plusieurs séparateurs (`\n\n`, `\n`, `. `, ` `) pour respecter la structure naturelle du texte.

6. **[`6_semantic_chunking.py`](6_semantic_chunking.py)** — *Chunking sémantique*
   - Utilise `SemanticChunker` et les embeddings `gemini-embedding-001` pour regrouper les phrases par **sens** plutôt que par nombre de caractères.

7. **[`7_agentic_chunking.py`](7_agentic_chunking.py)** — *Chunking agentique (piloté par LLM)*
   - Demande directement au LLM (`gemini-3.5-flash-lite`) d'identifier les frontières thématiques naturelles et d'insérer des marqueurs de découpage (`<<<SPLIT>>>`).

---

## Exécution rapide

```bash
# 1. Créer la base vectorielle à partir de docs/
python 1_ingestion_pipeline.py

# 2. Tester la recherche de documents
python 2_retrieval_pipeline.py

# 3. Générer une réponse avec Gemini
python 3_answer_generation.py

# 4. Lancer le chatbot RAG avec historique
python 4_history_aware_generation.py

# 5-7. Explorer les stratégies de chunking
python 5_recursive_character_text_splitter.py
python 6_semantic_chunking.py
python 7_agentic_chunking.py
```
