# Guide d'installation — Workshop RAG

Suivez ces étapes pas à pas pour préparer votre environnement virtuel Python (`venv`) et installer toutes les dépendances nécessaires au workshop.

---

## Étape 1 — Vérifier Python (3.10 ou plus récent)

Ouvrez un terminal et vérifiez que Python est bien installé :

- **Windows :**
  ```powershell
  python --version
  ```
- **macOS / Linux :**
  ```bash
  python3 --version
  ```

> Si Python n'est pas installé, téléchargez-le sur [python.org/downloads](https://www.python.org/downloads/).  
> **Important sur Windows :** cochez la case **"Add python.exe to PATH"** lors de l'installation.

---

## Étape 2 — Cloner le projet

```bash
git clone https://github.com/rayen-bou9a/RAG-session.git
cd RAG-session
```

---

## Étape 3 — Créer l'environnement virtuel (`venv`)

Dans le dossier du projet, créez un environnement virtuel nommé `.venv` :

- **Windows :**
  ```powershell
  python -m venv .venv
  ```
- **macOS / Linux :**
  ```bash
  python3 -m venv .venv
  ```

---

## Étape 4 — Activer l'environnement virtuel

Activez `.venv` selon votre système :

- **Windows (PowerShell) :**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  > Si PowerShell bloque le script avec une erreur de sécurité, exécutez une seule fois :
  > ```powershell
  > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  > ```
  > puis relancez `.\.venv\Scripts\Activate.ps1`.

- **Windows (Invite de commandes / CMD) :**
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

- **macOS / Linux :**
  ```bash
  source .venv/bin/activate
  ```

✅ **Vérification :** vous devez voir `(.venv)` au début de la ligne de votre terminal.

---

## Étape 5 — Installer les dépendances

Avec `(.venv)` activé, mettez à jour `pip` puis installez les paquets du projet :

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Étape 6 — Configurer votre clé API Gemini (gratuite)

1. Rendez-vous sur [Google AI Studio](https://aistudio.google.com/apikey) et créez une clé API gratuite.
2. Créez votre fichier `.env` à partir du modèle `.env.example` :
   - **Windows (PowerShell / CMD) :**
     ```powershell
     copy .env.example .env
     ```
   - **macOS / Linux :**
     ```bash
     cp .env.example .env
     ```
3. Ouvrez le fichier `.env` et remplacez `your_gemini_api_key_here` par votre clé :
   ```env
   GOOGLE_API_KEY=votre_cle_api_ici
   ```

---

## Étape 7 — Vérifier que tout fonctionne

1. **Test rapide sans clé API :**
   ```bash
   python 5_recursive_character_text_splitter.py
   ```
2. **Test complet avec Gemini + ChromaDB :**
   ```bash
   python 1_ingestion_pipeline.py
   python 2_retrieval_pipeline.py
   python 3_answer_generation.py
   ```

---

## Quitter l'environnement virtuel

Lorsque vous avez terminé le workshop, vous pouvez désactiver l'environnement virtuel :

```bash
deactivate
```
