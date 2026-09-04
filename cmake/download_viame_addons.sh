#!/bin/sh

# Usage: download_viame_addons.sh [ADDON-NAME ...]
# With no names, every add-on listed in download_viame_addons.csv is installed.

export VIAME_INSTALL=${VIAME_INSTALL:-/opt/noaa/viame}
export DOWNLOAD_LOCATION=/tmp/VIAME-Addons

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Ensure Download Location is Created
mkdir -p ${DOWNLOAD_LOCATION}

# Download Optional Packages
while IFS=, read -r ADDON_NAME DOWNLOAD_URL DESCRIPTION
do
  if [ $# -gt 0 ]; then
    case " $* " in
      *" ${ADDON_NAME} "*) ;;
      *) continue ;;
    esac
  fi
  wget -O "${DOWNLOAD_LOCATION}/${ADDON_NAME}.zip" ${DOWNLOAD_URL}
  unzip -o "${DOWNLOAD_LOCATION}/${ADDON_NAME}.zip" -d ${VIAME_INSTALL}
done < "${SCRIPT_DIR}/download_viame_addons.csv"

# Ensure Download Location is Removed
rm -rf ${DOWNLOAD_LOCATION}
