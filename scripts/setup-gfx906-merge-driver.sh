#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_root"

git config merge.keep-gfx906.name "Keep local gfx906 integration files"
git config merge.keep-gfx906.driver true
git config pull.rebase false
git config rerere.enabled true

echo "Configured merge.keep-gfx906 in $repo_root"
echo "Protected gfx906 files will now keep the local version on merge/pull conflicts."
echo "Use scripts/gfx906-safe-pull.sh for the safest workflow."