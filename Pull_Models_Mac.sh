#!/bin/bash

# Nom du modele LLM
LLM_MODEL_NAME="llama3.1:8b"

# Nom du modele d'embedding
EMB_MODEL_NAME="bge-m3"

# Chargement du modele LLM
echo "Pull du modelle de LLM : $LLM_MODEL_NAME"
ollama pull $LLM_MODEL_NAME

# Chargement du modele d'embedding
echo "Pull du modele d'embedding : $EMB_MODEL_NAME"
ollama pull $EMB_MODEL_NAME

# Libraires python
echo "Installation des libraires python (peut prendre un certain temps)..."
pip install -r requirements.txt* 2>/dev/null
echo "Installation des libraires terminé."

# Message final
echo "Vous pouvez lancer le serveur !"