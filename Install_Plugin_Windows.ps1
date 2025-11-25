# Nom du plugin
$PluginName = "QuestForgeAI"

# Dossier des moteurs
$UEBaseDir = "C:\Program Files\Epic Games"

# Récupère toutes les versions UE
$UEVersions = Get-ChildItem -Directory "$UEBaseDir\UE_*" -ErrorAction SilentlyContinue |
              Sort-Object Name

if ($UEVersions.Count -eq 0) {
    Write-Output "Aucune installation Unreal Engine trouvée dans $UEBaseDir."
    exit 1
}

# Version la plus récente
$LatestUE = $UEVersions[-1].FullName

Write-Output "Version Unreal la plus récente détectée : $LatestUE"

# Dossier Plugins du moteur
$TargetDir = "$LatestUE\Engine\Plugins"

# Crée dossier si nécessaire
if (!(Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}

# Copie du plugin
Copy-Item -Path ".\$PluginName" -Destination $TargetDir -Recurse -Force

Write-Output "Plugin '$PluginName' installé dans $TargetDir"

###############################################################################
################### MODIFICATION DU FICHIER BaseEngine.ini ####################
###############################################################################

$BaseEngineIni = Join-Path $LatestUE "Engine\Config\BaseEngine.ini"

if (-not (Test-Path $BaseEngineIni)) {
    Write-Host "⚠ BaseEngine.ini introuvable : $BaseEngineIni"
    exit 1
}

Write-Host " Modification de HttpActivityTimeout dans BaseEngine.ini..."

# Backup
$BackupFile = "$BaseEngineIni.bak"
Copy-Item $BaseEngineIni $BackupFile -Force

# Charger contenu
$Content = Get-Content $BaseEngineIni

# Remplacer uniquement la bonne ligne
$Content = $Content -replace "HttpActivityTimeout=30", "HttpActivityTimeout=300"

# Re-écrire fichier
Set-Content -LiteralPath $BaseEngineIni -Value $Content

Write-Host "✔ HttpActivityTimeout mis à 300"
Write-Host "✔ Backup créé : $BackupFile"
