# Nom du modele LLM
$ModelNameLLM = "llama3.1:8b"

# Nom du modele d'embedding
$ModelNameEmb = "bge-m3"

# Chargement du modele LLM
Write-Host "Pull du modelle de LLM : $ModelNameLLM"
ollama pull $ModelNameLLM

# Chargement du modele d'embedding
Write-Host "Pull du modele d'embedding : $ModelNameEmb"
ollama pull $ModelNameEmb

# Libraires python
Write-Host "Installation des libraires python (peut prendre un certain temps)..."
pip install -r requirements.txt *>$null
Write-Host "Installation des libraires terminé."

# Message final
Write-Host "Vous pouvez lancer le serveur !"
