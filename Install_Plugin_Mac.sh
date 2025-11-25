#!/bin/bash

# Nom du plugin (dossier)
PLUGIN_NAME="QuestForgeAI"

# Dossier racine où Epic Games installe les moteurs
UE_BASE_DIR="/Users/Shared/Epic Games"

# Récupère la dernière version UE directement (sans échappement interne)
LATEST_UE=$(ls -d "$UE_BASE_DIR"/UE_* 2>/dev/null | sort -V | tail -n 1)

# Vérifie si une version a été trouvée
if [ -z "$LATEST_UE" ]; then
    echo "Aucune installation Unreal Engine trouvée dans $UE_BASE_DIR."
    exit 1
fi

# Sélectionne la version la plus récente
echo "Version Unreal la plus récente trouvée : $LATEST_UE"

# Dossier où copier le plugin
TARGET_DIR="$LATEST_UE/Engine/Plugins"

# Crée le dossier si besoin
mkdir -p "$TARGET_DIR"

# Copie le plugin
cp -R "./$PLUGIN_NAME" "$TARGET_DIR"

echo "Plugin '$PLUGIN_NAME' installé dans $TARGET_DIR"

###############################################################################
# MODIFICATION DE BaseEngine.ini
###############################################################################

BASE_ENGINE_INI="$LATEST_UE/Engine/Config/BaseEngine.ini"

if [ ! -f "$BASE_ENGINE_INI" ]; then
    echo "⚠️  BaseEngine.ini introuvable : $BASE_ENGINE_INI"
    exit 1
fi

echo "Modification de HttpActivityTimeout dans BaseEngine.ini..."

# Sauvegarde
cp "$BASE_ENGINE_INI" "$BASE_ENGINE_INI.bak"

# Remplacement sécurisé
sed -i '' 's/HttpActivityTimeout=30/HttpActivityTimeout=300/' "$BASE_ENGINE_INI"

echo "HttpActivityTimeout modifié à 300 avec succès."
echo "Backup créé : $BASE_ENGINE_INI.bak"