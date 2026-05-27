#!/bin/bash

# get absolute directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# get directories
chtc=$(basename "$SCRIPT_DIR")
project_dir=$(basename "$(dirname "$SCRIPT_DIR")")
user_dir=$(basename "$(dirname "$(dirname "$SCRIPT_DIR")")")

# debug output
# echo "CHTC folder: $chtc"
echo "Project folder: $project_dir"
echo "User folder: $user_dir"



# Target directory
TARGET="data"
mkdir -p "$TARGET"
FORCE=false

# Process arguments
for arg in "$@"; do
    if [ "$arg" == "-f" ]; then
        FORCE=true
    else
        echo "Warning: Ignoring invalid directory '$arg'"
    fi
done

# Find all "data" directories inside ./runs
find /staging/$user_dir/$project_dir -type d -name "data" | while read -r src; do
  echo "Processing: $src"
  
  # Loop through files and subfolders inside each source "data"
  find "$src" -mindepth 1 -maxdepth 1 | while read -r item; do
    base=$(basename "$item")
    dest="$TARGET/$base"

    if [ -e "$dest" ] && [ "$FORCE" = false ]; then
      echo "⚠️  Skipping existing: $dest"
      continue
    fi

    if [ -e "$dest" ] && [ "$FORCE" = true ]; then
      echo "⚠️  Overwriting: $dest"
      rm -rf "$dest"
    fi

    echo "Moving: $item -> $dest"
    mv "$item" "$dest"
  done

  # Remove empty source "data" directory
  rmdir "$src" 2>/dev/null || true
done

echo "✅ Done."