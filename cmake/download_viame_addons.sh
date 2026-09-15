#!/bin/sh

# Usage: download_viame_addons.sh [ADDON-NAME ...]
# With no names, every add-on listed in download_viame_addons.csv is installed.
#
# The shell fallback for `viame add-ons install`, for an install whose python
# is not set up yet. It downloads the way the configure step does
# (NormalizeDownloadUrl and DownloadFile in cmake/common_macros.cmake): a
# Google Drive share link becomes its direct download, and an archive whose
# MD5 is not the one the CSV gives is not unpacked.

export VIAME_INSTALL=${VIAME_INSTALL:-/opt/noaa/viame}
export DOWNLOAD_LOCATION=/tmp/VIAME-Addons

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# A Drive link like https://drive.google.com/file/d/<ID>/view?usp=sharing is
# an HTML page, not the file; confirm=t skips the page Google serves instead
# of large files.
normalize_url() {
  id=$(printf '%s' "$1" | sed -n \
    -e 's#.*drive\.google\.com/file/d/\([A-Za-z0-9_-]*\).*#\1#p' \
    -e 's#.*drive\.google\.com/\(uc\|open\)?.*id=\([A-Za-z0-9_-]*\).*#\2#p' | head -n 1)
  if [ -n "$id" ]; then
    printf 'https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t' "$id"
  else
    printf '%s' "$1"
  fi
}

if [ "${1:-}" = "--print-url" ]; then
  normalize_url "$2"
  echo
  exit 0
fi

mkdir -p "${DOWNLOAD_LOCATION}"

status=0

# name, url, description, md5, platform, enable flags
while IFS=, read -r ADDON_NAME DOWNLOAD_URL DESCRIPTION MD5 REST
do
  ADDON_NAME=$(printf '%s' "${ADDON_NAME}" | tr -d ' ')
  DOWNLOAD_URL=$(printf '%s' "${DOWNLOAD_URL}" | tr -d ' ')
  MD5=$(printf '%s' "${MD5}" | tr -d ' ')

  [ -n "${ADDON_NAME}" ] || continue

  if [ $# -gt 0 ]; then
    case " $* " in
      *" ${ADDON_NAME} "*) ;;
      *) continue ;;
    esac
  fi

  archive="${DOWNLOAD_LOCATION}/${ADDON_NAME}.zip"

  if ! wget -O "${archive}" "$(normalize_url "${DOWNLOAD_URL}")"; then
    echo "Could not download ${ADDON_NAME}" >&2
    status=1
    continue
  fi

  if [ -n "${MD5}" ] && command -v md5sum > /dev/null 2>&1; then
    actual=$(md5sum "${archive}" | cut -d' ' -f1)
    if [ "${actual}" != "${MD5}" ]; then
      echo "${ADDON_NAME}: MD5 ${actual} is not the expected ${MD5}; not installed" >&2
      status=1
      continue
    fi
  fi

  if ! unzip -o "${archive}" -d "${VIAME_INSTALL}"; then
    echo "Could not unpack ${ADDON_NAME}" >&2
    status=1
  fi
done < "${SCRIPT_DIR}/download_viame_addons.csv"

rm -rf "${DOWNLOAD_LOCATION}"

exit ${status}
