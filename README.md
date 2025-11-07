# 🧠 Serveur RAG (Retrieval-Augmented Generation)

Implémentation d'un serveur LLM (`RAG`) en Python.

---

## 📂 Structure

```
📁 RAG/
│   ├── init_rag.py         # Script d'initialisation du moteur RAG
│   ├── run.py              # Run serveur avec uvicorn
│   └── server_rag.py       # FastAPI avec endpoint POST /query
│
📁 LOGS/
│   └── SERVER_MISTRAL/     # Dossier où sont enregistrés les logs d’exécution et réponses
│
📁 DATA/                    # Documents support du rag (.pdf, .txt, .csv)
│
runRAG.bat                 # Script pour lancer le serveur sur Windows
requirements.txt           # Liste des dépendances Python
README.md                 
```

---

## ⚙️ Fonctionnement

1. **Chargement des documents**
   - Les fichiers dans `DATA/` sont chargés.
   - Les extensions prises en charge sont : `.txt`, `.pdf`, `.csv`.

2. **Découpage en chunks**
   - Les documents sont découpés en blocs de K chunks de taille fixée.

3. **Vectorisation**
   - Chaque chunk est converti en vecteur via un modèle d'embedding choisi.

4. **Indexation FAISS**
   - Tous les vecteurs sont stockés dans un index FAISS (peut prendre du temps).

5. **Requête**
   - L'utilisateur envoie une question en POST via l'API.
   - Le système interroge les chunks les plus pertinents et génère une réponse à l'aide du modèle Ollama (`mistral-small` par défaut).

---

## 🚀 Installation et démarrage

### 1. Prérequis

- Python 3.10+
- [Ollama](https://ollama.com/) installé localement
- Modèle Ollama téléchargé (`mistral-small:24b`, ou autre)
- Modèle pour l'EMBEDDING (modèles disponibles sur https://huggingface.co/models?other=text-embeddings-inference)

Exemples :
- BAAI/bge-m3
- nomic-ai/nomic-embed-text-v1.5

Télécharger un modèle sur Ollama, via terminal :
```bash
ollama pull mistral-small
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer le serveur

#### Via le script Windows `runRAG.bat` :
```bat
@echo off
echo ===============================
echo Starting RAG server on port 8000
echo ===============================
python RAG/run.py --modelembedding "nomic-ai/nomic-embed-text-v1.5" --kchunk 4 --sizechunk 4000
pause
```

Ce script utilise :
- Modèle d'embedding : `nomic-ai/nomic-embed-text-v1.5`
- Nombre de chunks pertinents à extraire (`kchunk`) : 4
- Taille maximale d’un chunk (`sizechunk`) : 4000 caractères

*Les modèles d'embedding sont téléchargés localement puis les chunks sont découpés et sauvegardés localement également. 
Nécessite du temps selon le découpage et le modèle choisi (plusieurs minutes voire heures).*

---

## 🧪 Utilisation de l’API

### 🔗 Endpoint

```http
POST /query
```

### 🔍 Corps de la requête

```json
{
  "prompt": "Quelles sont les informations à la date du 21 mai 1944 ?"
}
```

### ✅ Réponse

```json
{
  "answer": "Texte de réponse généré à partir des documents.",
  "execution_time_sec": 1.58
}
```

---

## 📦 Exemple de test avec `curl`

```bash
curl -X POST http://127.0.0.1:8000/query \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"Quelles sont les informations à la date du 21 mai 1944 ?\"}"
```

---
