#!/bin/bash
# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

# Script to add or replace SSH remotes for git submodules in the packages directory.
#
# Usage:
#   ./use_ssh_submodule_links.sh add      - Add SSH remote named "ssh" alongside origin
#   ./use_ssh_submodule_links.sh replace  - Replace origin URL with SSH URL

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGES_DIR="$(dirname "$SCRIPT_DIR")/packages"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <add|replace>"
    echo "  add     - Add SSH remote named 'ssh' alongside existing origin"
    echo "  replace - Replace origin URL with SSH URL"
    exit 1
fi

MODE="$1"

if [ "$MODE" != "add" ] && [ "$MODE" != "replace" ]; then
    echo "Error: Invalid mode '$MODE'. Use 'add' or 'replace'."
    exit 1
fi

# Function to convert HTTPS URL to SSH URL
https_to_ssh() {
    local url="$1"
    # Convert https://github.com/org/repo.git to git@github.com:org/repo.git
    # Also handle URLs without .git suffix
    echo "$url" | sed -E 's|https://([^/]+)/(.+)|git@\1:\2|'
}

# Function to process a single git repository
process_repo() {
    local repo_path="$1"
    local repo_name="$(basename "$repo_path")"

    if [ ! -d "$repo_path/.git" ] && [ ! -f "$repo_path/.git" ]; then
        return 0
    fi

    cd "$repo_path"

    # Get the origin URL
    local origin_url
    origin_url=$(git remote get-url origin 2>/dev/null) || {
        echo "  Skipping $repo_name: no origin remote"
        return 0
    }

    # Check if it's an HTTPS URL
    if [[ ! "$origin_url" =~ ^https:// ]]; then
        echo "  Skipping $repo_name: origin is not HTTPS ($origin_url)"
        return 0
    fi

    # Convert to SSH URL
    local ssh_url
    ssh_url=$(https_to_ssh "$origin_url")

    if [ "$MODE" == "add" ]; then
        # Check if ssh remote already exists
        if git remote get-url ssh &>/dev/null; then
            echo "  $repo_name: ssh remote already exists, updating URL"
            git remote set-url ssh "$ssh_url"
        else
            echo "  $repo_name: adding ssh remote -> $ssh_url"
            git remote add ssh "$ssh_url"
        fi
    else
        # Replace origin
        echo "  $repo_name: replacing origin -> $ssh_url"
        git remote set-url origin "$ssh_url"
    fi
}

# Function to recursively find and process git repositories
process_directory() {
    local dir="$1"
    local indent="$2"
    local skip_children="$3"

    for item in "$dir"/*; do
        if [ -d "$item" ]; then
            local name="$(basename "$item")"

            # Skip downloads and patches directories
            if [ "$name" == "downloads" ] || [ "$name" == "patches" ]; then
                continue
            fi

            # Check if this is a git repository
            if [ -d "$item/.git" ] || [ -f "$item/.git" ]; then
                process_repo "$item"
            fi

            # Skip recursing into children if requested (e.g., pytorch subprojects)
            if [ "$skip_children" == "true" ]; then
                continue
            fi

            # Check if this is the pytorch directory - process it but skip its children
            if [ "$name" == "pytorch" ]; then
                process_directory "$item" "  $indent" "true"
            else
                # Recurse into subdirectories to find nested submodules
                process_directory "$item" "  $indent" "false"
            fi
        fi
    done
}

echo "Processing submodules in: $PACKAGES_DIR"
echo "Mode: $MODE"
echo ""

if [ ! -d "$PACKAGES_DIR" ]; then
    echo "Error: Packages directory not found: $PACKAGES_DIR"
    exit 1
fi

process_directory "$PACKAGES_DIR" "" "false"

echo ""
echo "Done!"
