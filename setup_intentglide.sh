#!/usr/bin/env bash
set -euo pipefail

# Copy this scaffold into an existing repository without overwriting files.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-.}"
TARGET="$(cd "$TARGET" && pwd)"

if [[ "$TARGET" == "$ROOT" ]]; then
  echo "Project files are already in: $TARGET"
  exit 0
fi

while IFS= read -r -d '' src; do
  rel="${src#"$ROOT"/}"
  [[ "$rel" == "README.md" ]] && continue
  mkdir -p "$TARGET/$(dirname "$rel")"
  if [[ -e "$TARGET/$rel" ]]; then
    echo "Keeping existing: $rel"
  else
    cp "$src" "$TARGET/$rel"
    echo "Added: $rel"
  fi
done < <(find "$ROOT" -type f -print0)

echo
echo "Scaffold copied. Review changes, then run:"
echo "  git add ."
echo "  git commit -m \"chore: scaffold IntentGlide modules\""
echo "  git push origin main"
