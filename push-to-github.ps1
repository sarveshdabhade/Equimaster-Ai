# Equimaster-Ai: Add, commit, and push to GitHub
# Run this in Cursor terminal (Ctrl+`) or PowerShell. Replace YOUR_USERNAME with your GitHub username.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Clean stale Git locks
Remove-Item ".git\index.lock" -Force -ErrorAction SilentlyContinue
Get-ChildItem ".git\objects\27\tmp_obj_*" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

# Stage and commit
git add .
git commit -m "Day 1: Initial Setup"

# Add GitHub remote and push (replace YOUR_USERNAME with your GitHub username)
$repoUrl = "https://github.com/YOUR_USERNAME/Equimaster-Ai.git"
git remote remove origin 2>$null
git remote add origin $repoUrl
git branch -M main
git push -u origin main

Write-Host "Done. Your repo is at: https://github.com/YOUR_USERNAME/Equimaster-Ai" -ForegroundColor Green
