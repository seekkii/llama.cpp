#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$repo_root/scripts/setup-gfx906-merge-driver.sh" >/dev/null

cd "$repo_root"
git pull --no-rebase "$@"