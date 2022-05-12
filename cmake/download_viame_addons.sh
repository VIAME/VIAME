#!/bin/sh

# Configurable Input Paths
export VIAME_INSTALL=/opt/noaa/viame
export DOWNLOAD_LOCATION=/tmp/VIAME-Addons

# Ensure Download Location is Created
mkdir -p ${DOWNLOAD_LOCATION}

# Download All Optional Packages
while IFS=, read -r ADDON_NAME DOWNLOAD_URL DESCRIPTION
do
  wget -O "${DOWNLOAD_LOCATION}/${ADDON_NAME}.zip" ${DOWNLOAD_URL}
  unzip -o "${DOWNLOAD_LOCATION}/${ADDON_NAME}.zip" -d ${VIAME_INSTALL}
done < download_viame_addons.csv

# Ensure Download Location is Removed
rm -rf ${DOWNLOAD_LOCATION}
