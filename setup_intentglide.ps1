param(
    [Parameter(Mandatory=$true)]
    [string]$TargetRepository
)

$ErrorActionPreference = "Stop"
$SourceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TargetRoot = (Resolve-Path $TargetRepository).Path

Get-ChildItem -LiteralPath $SourceRoot -Recurse -File | ForEach-Object {
    $relative = $_.FullName.Substring($SourceRoot.Length).TrimStart('\','/')
    if ($relative -eq "README.md") { return }

    $destination = Join-Path $TargetRoot $relative
    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null

    if (Test-Path -LiteralPath $destination) {
        Write-Host "Keeping existing: $relative"
    } else {
        Copy-Item -LiteralPath $_.FullName -Destination $destination
        Write-Host "Added: $relative"
    }
}

Write-Host ""
Write-Host "Scaffold copied. Review changes, then run:"
Write-Host "  git add ."
Write-Host '  git commit -m "chore: scaffold IntentGlide modules"'
Write-Host "  git push origin main"
