#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/push_to_github.sh choi0312 deepguard-multitask-deepfake-detection
#
# Before running:
#   1. Create an empty GitHub repository with the same name, or
#   2. Install GitHub CLI and run: gh auth login

GITHUB_USER=${1:-choi0312}
REPO_NAME=${2:-deepguard-multitask-deepfake-detection}
REMOTE_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

if [ ! -d .git ]; then
  git init
fi

git add .
git commit -m "Refactor DeepGuard into reproducible multi-task deepfake pipeline" || true
git branch -M main

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REMOTE_URL"
else
  git remote set-url origin "$REMOTE_URL"
fi

git push -u origin main
