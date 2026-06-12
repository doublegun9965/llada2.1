param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Message
)

$ErrorActionPreference = "Stop"

git status --short

$changes = git status --porcelain
if (-not $changes) {
    Write-Host "No changes to commit."
    exit 0
}

git add -A
git commit -m $Message
git push
