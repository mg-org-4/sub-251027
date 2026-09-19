param(
    [string]$ComfyUIRoot = "D:\StabilityMatrix\Data\Packages\ComfyUI_W"
)

$ErrorActionPreference = "Stop"
$source = (Resolve-Path $PSScriptRoot).Path
$customNodes = Join-Path $ComfyUIRoot "custom_nodes"
$destination = Join-Path $customNodes "ComfyUI-H3-Continuum"
$legacyDestination = Join-Path $customNodes "ComfyUI-H3-Continuum-Join"
$backupRoot = Join-Path $ComfyUIRoot "h3_continuum_backups"
$registryManifest = Join-Path $source "REGISTRY_MANIFEST.sha256"

if (-not (Test-Path $ComfyUIRoot -PathType Container)) {
    throw "ComfyUI root not found: $ComfyUIRoot"
}
if (-not (Test-Path (Join-Path $ComfyUIRoot "comfy") -PathType Container)) {
    throw "The selected folder does not look like a ComfyUI installation: $ComfyUIRoot"
}
New-Item -ItemType Directory -Force -Path $customNodes | Out-Null

if ([System.StringComparer]::OrdinalIgnoreCase.Equals($source, $destination)) {
    Write-Host "Already installed at: $destination"
    exit 0
}

$sourceRoot = [IO.Path]::GetFullPath($source).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
$destinationRoot = [IO.Path]::GetFullPath($destination).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
if (-not (Test-Path -LiteralPath $registryManifest -PathType Leaf)) {
    throw "Registry manifest not found: $registryManifest"
}

$filesToInstall = foreach ($line in Get-Content -LiteralPath $registryManifest) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line -notmatch '^(?<Hash>[0-9a-fA-F]{64})\s{2}(?<Path>.+)$') {
        throw "Invalid Registry manifest line: $line"
    }
    $relativePath = $Matches.Path.Replace('/', [IO.Path]::DirectorySeparatorChar)
    if ([IO.Path]::IsPathRooted($relativePath)) {
        throw "Registry manifest contains an absolute path: $relativePath"
    }
    $sourceFile = [IO.Path]::GetFullPath((Join-Path $source $relativePath))
    if (-not $sourceFile.StartsWith($sourceRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Registry manifest path escaped the source directory: $relativePath"
    }
    if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
        throw "Registry manifest file is missing: $relativePath"
    }
    $actualHash = (Get-FileHash -LiteralPath $sourceFile -Algorithm SHA256).Hash
    if (-not $actualHash.Equals($Matches.Hash, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Registry manifest hash mismatch: $relativePath"
    }
    [pscustomobject]@{
        RelativePath = $relativePath
        Source = $sourceFile
    }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
foreach ($old in @($destination, $legacyDestination)) {
    if ((Test-Path $old) -and -not [System.StringComparer]::OrdinalIgnoreCase.Equals($source, $old)) {
        $backupName = "$(Split-Path -Leaf $old).backup-$stamp"
        $backup = Join-Path $backupRoot $backupName
        Move-Item -LiteralPath $old -Destination $backup
        Write-Host "Existing installation moved to: $backup"
    }
}

New-Item -ItemType Directory -Force -Path $destination | Out-Null
foreach ($file in $filesToInstall) {
    $destinationFile = [IO.Path]::GetFullPath((Join-Path $destination $file.RelativePath))
    if (-not $destinationFile.StartsWith($destinationRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Registry manifest path escaped the destination directory: $($file.RelativePath)"
    }
    $parent = Split-Path -Parent $destinationFile
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Copy-Item -LiteralPath $file.Source -Destination $destinationFile -Force
}
Copy-Item -LiteralPath $registryManifest -Destination (Join-Path $destination "REGISTRY_MANIFEST.sha256") -Force

$python = Join-Path $ComfyUIRoot "venv\Scripts\python.exe"
$verifier = Join-Path $destination "tools\verify_runtime.py"
if (Test-Path $python) {
    if (-not (Test-Path $verifier -PathType Leaf)) {
        throw "Runtime verifier is missing from the installed package: $verifier"
    }
    & $python $verifier --comfy-root $ComfyUIRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime verification failed. See the output above."
    }
} else {
    Write-Warning "venv Python was not found; installation was copied but runtime verification was skipped."
}

Write-Host "Installed H3 Continuum to: $destination"
Write-Host "Restart ComfyUI. If the browser cached the old frontend, use a hard reload."
