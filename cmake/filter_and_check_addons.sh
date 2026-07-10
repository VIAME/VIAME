#!/bin/bash

# Filter and Check Add-ons Script
#
# Compares pipelines in downloaded add-on archives against the source
# configs/add-ons/ tree. If differences are found, prompts the user
# to select add-ons to update, then creates new zip files with the
# latest pipelines (from source) and models (from old archive), with
# a bumped minor version number.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ADDONS_DIR="$SRC_DIR/configs/add-ons"
DOWNLOADS_DIR="$SRC_DIR/packages/downloads"
CSV_FILE="$SCRIPT_DIR/download_viame_addons.csv"
CACHE_FILE="$SRC_DIR/build/CMakeCache.txt"

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Collect list of add-on names from CSV and/or downloads folder
declare -A ADDON_ARCHIVES

# Check CMakeCache for enabled add-ons and their download paths
if [ -f "$CACHE_FILE" ]; then
  echo "Reading enabled add-ons from CMakeCache..."
  while IFS= read -r line; do
    if [[ "$line" =~ ^VIAME_DOWNLOAD_MODELS-([A-Z0-9_-]+):BOOL=ON$ ]]; then
      name="${BASH_REMATCH[1]}"
      archive="$DOWNLOADS_DIR/VIAME-${name}-Models.zip"
      if [ -f "$archive" ]; then
        ADDON_ARCHIVES["$name"]="$archive"
      fi
    fi
  done < "$CACHE_FILE"
fi

# Also scan downloads folder for any VIAME-*-Models.zip files
for archive in "$DOWNLOADS_DIR"/VIAME-*-Models.zip; do
  [ -f "$archive" ] || continue
  basename=$(basename "$archive")
  # Extract name: VIAME-FOO-BAR-Models.zip -> FOO-BAR
  name=$(echo "$basename" | sed 's/^VIAME-//;s/-Models\.zip$//')
  if [ -n "$name" ]; then
    ADDON_ARCHIVES["$name"]="$archive"
  fi
done

if [ ${#ADDON_ARCHIVES[@]} -eq 0 ]; then
  echo "No downloaded add-on archives found."
  echo "Checked: $DOWNLOADS_DIR for VIAME-*-Models.zip files"
  [ -f "$CACHE_FILE" ] && echo "Checked: $CACHE_FILE for enabled add-ons"
  exit 0
fi

echo "Found ${#ADDON_ARCHIVES[@]} downloaded add-on archive(s)."
echo ""

# Convert CSV add-on name (uppercase) to folder name (lowercase)
addon_name_to_folder() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

# Compare pipeline files between archive and source add-on folder
# Returns 0 if differences found, 1 if identical or no comparison possible
declare -A DIFF_ADDONS
declare -A DIFF_DETAILS

for name in $(echo "${!ADDON_ARCHIVES[@]}" | tr ' ' '\n' | sort); do
  archive="${ADDON_ARCHIVES[$name]}"
  folder_name=$(addon_name_to_folder "$name")
  addon_folder="$ADDONS_DIR/$folder_name"

  if [ ! -d "$addon_folder" ]; then
    echo "[$name] No source folder at $addon_folder - skipping"
    continue
  fi

  echo -n "[$name] Comparing pipelines... "

  # Extract pipeline file list from the archive (exclude models/ and directories)
  archive_pipes=$(unzip -l "$archive" 2>/dev/null | \
    awk '/configs\/pipelines\/[^\/]+\.(pipe|conf)$/ {print $NF}' | \
    sed 's|configs/pipelines/||' | sort)

  # Get pipeline file list from source add-on folder (top-level .pipe and .conf only)
  source_pipes=$(find "$addon_folder" -maxdepth 1 \( -name "*.pipe" -o -name "*.conf" \) \
    -printf '%f\n' | sort)

  if [ -z "$archive_pipes" ] && [ -z "$source_pipes" ]; then
    echo "no pipeline files in either location"
    continue
  fi

  # Extract archive pipeline files to temp for comparison
  extract_dir="$TEMP_DIR/extract_$name"
  mkdir -p "$extract_dir"
  unzip -q -o "$archive" 'configs/pipelines/*.pipe' 'configs/pipelines/*.conf' \
    -d "$extract_dir" 2>/dev/null || true

  archive_pipe_dir="$extract_dir/configs/pipelines"
  has_diff=false
  diff_files=""

  # Compare files that exist in both locations
  all_pipes=$(echo -e "${archive_pipes}\n${source_pipes}" | sort -u | grep -v '^$' || true)

  for pipe_file in $all_pipes; do
    archive_file="$archive_pipe_dir/$pipe_file"
    source_file="$addon_folder/$pipe_file"

    if [ ! -f "$archive_file" ] && [ -f "$source_file" ]; then
      has_diff=true
      diff_files="$diff_files  + $pipe_file (new in source, not in archive)\n"
    elif [ -f "$archive_file" ] && [ ! -f "$source_file" ]; then
      has_diff=true
      diff_files="$diff_files  - $pipe_file (in archive, not in source)\n"
    elif [ -f "$archive_file" ] && [ -f "$source_file" ]; then
      if ! diff -q "$archive_file" "$source_file" > /dev/null 2>&1; then
        has_diff=true
        diff_files="$diff_files  ~ $pipe_file (modified)\n"
      fi
    fi
  done

  if $has_diff; then
    echo "DIFFERENCES FOUND"
    echo -e "$diff_files"
    DIFF_ADDONS["$name"]="$archive"
    DIFF_DETAILS["$name"]="$diff_files"
  else
    echo "identical"
  fi
done

echo ""

if [ ${#DIFF_ADDONS[@]} -eq 0 ]; then
  echo "All add-on pipelines are up to date. Nothing to do."
  exit 0
fi

# Prompt user to select which add-ons to update
echo "========================================="
echo "The following add-ons have different pipelines:"
echo "========================================="

diff_names=($(echo "${!DIFF_ADDONS[@]}" | tr ' ' '\n' | sort))

for i in "${!diff_names[@]}"; do
  echo "  $((i+1)). ${diff_names[$i]}"
done
echo "  A. All of the above"
echo "  Q. Quit (do nothing)"
echo ""

read -p "Select add-on(s) to update (comma-separated numbers, A for all, Q to quit): " selection

if [[ "$selection" =~ ^[Qq]$ ]]; then
  echo "No changes made."
  exit 0
fi

selected_addons=()
if [[ "$selection" =~ ^[Aa]$ ]]; then
  selected_addons=("${diff_names[@]}")
else
  IFS=',' read -ra selections <<< "$selection"
  for sel in "${selections[@]}"; do
    sel=$(echo "$sel" | tr -d ' ')
    if [[ "$sel" =~ ^[0-9]+$ ]] && [ "$sel" -ge 1 ] && [ "$sel" -le ${#diff_names[@]} ]; then
      selected_addons+=("${diff_names[$((sel-1))]}")
    else
      echo "Warning: ignoring invalid selection '$sel'"
    fi
  done
fi

if [ ${#selected_addons[@]} -eq 0 ]; then
  echo "No valid selections made. Exiting."
  exit 0
fi

# Determine version from archive filename or default to 1.0
get_archive_version() {
  local archive_path="$1"
  local basename=$(basename "$archive_path")

  # Try to extract version from filename like VIAME-NAME-Models-v1.2.zip
  local version=$(echo "$basename" | grep -oP 'v(\d+\.\d+)' | head -1 | sed 's/^v//')

  if [ -z "$version" ]; then
    # Check CSV for a version hint, otherwise look at zip comment or default
    # Try to find version in the zip file contents
    local ver_file=$(unzip -l "$archive_path" 2>/dev/null | grep -oP 'v\d+\.\d+' | head -1 | sed 's/^v//')
    if [ -n "$ver_file" ]; then
      version="$ver_file"
    else
      version="1.0"
    fi
  fi

  echo "$version"
}

# Bump minor version: 1.1 -> 1.2, 2.3 -> 2.4
bump_minor_version() {
  local version="$1"
  local major=$(echo "$version" | cut -d. -f1)
  local minor=$(echo "$version" | cut -d. -f2)
  minor=$((minor + 1))
  echo "${major}.${minor}"
}

OUTPUT_DIR="$SRC_DIR/packages/downloads"

echo ""
echo "Creating updated add-on archives..."
echo ""

for name in "${selected_addons[@]}"; do
  archive="${DIFF_ADDONS[$name]}"
  folder_name=$(addon_name_to_folder "$name")
  addon_folder="$ADDONS_DIR/$folder_name"

  old_version=$(get_archive_version "$archive")
  new_version=$(bump_minor_version "$old_version")

  output_file="$OUTPUT_DIR/VIAME-${name}-Models-v${new_version}.zip"

  echo "[$name] Building updated archive (v${old_version} -> v${new_version})..."

  # Create staging directory with proper structure
  stage_dir="$TEMP_DIR/stage_$name"
  rm -rf "$stage_dir"
  mkdir -p "$stage_dir/configs/pipelines"

  # Extract models and transforms from old archive
  echo "  Extracting models from old archive..."
  unzip -q -o "$archive" 'configs/pipelines/models/*' -d "$stage_dir" 2>/dev/null || true
  unzip -q -o "$archive" 'configs/pipelines/transforms/*' -d "$stage_dir" 2>/dev/null || true

  # Copy latest pipeline files from source add-on folder
  echo "  Copying latest pipelines from source..."
  find "$addon_folder" -maxdepth 1 \( -name "*.pipe" -o -name "*.conf" \) \
    -exec cp {} "$stage_dir/configs/pipelines/" \;

  # Also copy subdirectories that contain pipeline files (embedded_single_stream, etc.)
  # but exclude 'models' and 'transforms' subdirs since those come from the archive
  for subdir in "$addon_folder"/*/; do
    [ -d "$subdir" ] || continue
    subdir_name=$(basename "$subdir")
    if [ "$subdir_name" != "models" ] && [ "$subdir_name" != "transforms" ]; then
      # Check if subdir contains .pipe or .conf files
      if find "$subdir" \( -name "*.pipe" -o -name "*.conf" \) -print -quit | grep -q .; then
        mkdir -p "$stage_dir/configs/pipelines/$subdir_name"
        find "$subdir" \( -name "*.pipe" -o -name "*.conf" \) \
          -exec cp {} "$stage_dir/configs/pipelines/$subdir_name/" \;
      fi
    fi
  done

  # Create the new zip
  echo "  Creating $output_file..."
  (cd "$stage_dir" && zip -q -r "$output_file" configs/)

  echo "  Done: $(basename "$output_file")"
  echo ""
done

echo "========================================="
echo "Updated add-on archive(s) created in:"
echo "  $OUTPUT_DIR"
echo "========================================="
