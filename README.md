# Workshop RAG (Retrieval-Augmented Generation) — Cours d'Algèbre Linéaire

Ce dépôt contient les supports et scripts pratiques pour le workshop sur le **RAG (Retrieval-Augmented Generation)** avec **LangChain**, **ChromaDB** et les **modèles gratuits de Google Gemini**, appliqués à un **cours d'algèbre linéaire sur les matrices**.

---

## Prérequis et installation

> **Pour les étudiants :** suivez le guide détaillé pas à pas dans **[`SETUP.md`](SETUP.md)** pour créer votre environnement virtuel (`venv`), installer les dépendances et configurer votre clé API Gemini.

### Modèles Gemini utilisés (Free Tier)

Configurés au même endroit dans [`config.py`](config.py) :

| Usage | Modèle Gemini | Classe LangChain |
|---|---|---|
| **Génération (LLM)** | `gemini-3.5-flash-lite` | `ChatGoogleGenerativeAI` |
| **Embeddings** | `gemini-embedding-001` | `GoogleGenerativeAIEmbeddings` |

---

## Organisation du projet

```text
RAG/
├── config.py                                # Configuration centrale (modèles Gemini, chemins, helpers)
├── docs/                                    # Base de connaissances : cours d'algèbre linéaire
│   ├── matrices_01_definitions_operations.txt
│   ├── matrices_02_determinant_inverse.txt
│   ├── matrices_03_systemes_lineaires.txt
│   └── matrices_04_valeurs_propres.txt
├── 1_ingestion_pipeline.py                  # Étape 1 : Chargement, découpage et indexation ChromaDB
├── 2_retrieval_pipeline.py                  # Étape 2 : Recherche vectorielle (top-k)
├── 3_answer_generation.py                   # Étape 3 : Génération augmentée (RAG complet)
├── 4_history_aware_generation.py            # Étape 4 : Chatbot RAG avec mémoire conversationnelle
├── 5_recursive_character_text_splitter.py   # Étape 5 : Découpage récursif
├── 6_semantic_chunking.py                   # Étape 6 : Chunking sémantique (par embeddings)
├── 7_agentic_chunking.py                    # Étape 7 : Chunking agentique (piloté par LLM)
├── SETUP.md                                 # Guide pas à pas d'installation (venv + .env)
├── requirements.txt                         # Dépendances Python
└── .env.example                             # Modèle de configuration de la clé API
```

---

## Les 7 notions du Workshop

### Partie 1 — Pipeline RAG de base

1. **[`1_ingestion_pipeline.py`](1_ingestion_pipeline.py)** — *Ingestion des documents*
   - Charge les 4 chapitres du cours sur les matrices depuis `docs/`.
   - Découpe le cours en *chunks* avec chevauchement (`chunk_size=800`, `chunk_overlap=100`).
   - Génère les vecteurs (*embeddings*) via `gemini-embedding-001` et les stocke dans **ChromaDB** (`db/chroma_db`).

2. **[`2_retrieval_pipeline.py`](2_retrieval_pipeline.py)** — *Recherche vectorielle (Retrieval)*
   - Interroge ChromaDB par similarité cosinus pour retrouver les extraits du cours correspondant à une question d'étudiant.
   - Inclut 8 questions d'entraînement couvrant les 4 chapitres.

3. **[`3_answer_generation.py`](3_answer_generation.py)** — *Génération augmentée (RAG complet)*
   - Injecte les extraits retrouvés dans un prompt structuré et demande à `gemini-3.5-flash-lite` de répondre uniquement à partir du cours.

4. **[`4_history_aware_generation.py`](4_history_aware_generation.py)** — *RAG conversationnel avec mémoire*
   - Chatbot interactif dans le terminal qui conserve l'historique (`chat_history`).
   - Reformule automatiquement les questions de suivi (ex. *"Et quels sont ses vecteurs propres ?"*) en questions autonomes avant la recherche.

### Partie 2 — Stratégies avancées de Chunking

5. **[`5_recursive_character_text_splitter.py`](5_recursive_character_text_splitter.py)** — *Découpage récursif*
   - Compare `CharacterTextSplitter` et `RecursiveCharacterTextSplitter` sur un texte d'algèbre linéaire.

6. **[`6_semantic_chunking.py`](6_semantic_chunking.py)** — *Chunking sémantique*
   - Regroupe les phrases par **sens mathématique** (produit matriciel, déterminant/inverse, valeurs propres) grâce aux embeddings `gemini-embedding-001`.

7. **[`7_agentic_chunking.py`](7_agentic_chunking.py)** — *Chunking agentique (piloté par LLM)*
   - Demande directement à `gemini-3.5-flash-lite` d'insérer des marqueurs `<<<SPLIT>>>` aux frontières thématiques naturelles.

---

## Exécution rapide

```bash
# 1. Indexer le cours sur les matrices dans ChromaDB
python 1_ingestion_pipeline.py

# 2. Tester la recherche d'extraits pertinents
python 2_retrieval_pipeline.py

# 3. Générer une réponse RAG avec Gemini
python 3_answer_generation.py

# 4. Lancer le chatbot interactif sur le cours
python 4_history_aware_generation.py

# 5-7. Explorer les stratégies de chunking
python 5_recursive_character_text_splitter.py
python 6_semantic_chunking.py
python 7_agentic_chunking.py
```
