#!/bin/sh

# Configurable Input Paths
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/.."
export DOWNLOAD_LOCATION=~/VIAME-Addons

# Ensure Download Location is Created
mkdir -p ${DOWNLOAD_LOCATION}

# Download All Optional Packages

# Habcam -
wget -O ${DOWNLOAD_LOCATION}/download1.zip https://data.kitware.com/api/v1/item/6011e3452fa25629b91ade60/download
unzip -o ${DOWNLOAD_LOCATION}/download1.zip -d ${VIAME_INSTALL}

# SEFSC -
wget -O ${DOWNLOAD_LOCATION}/download2.zip https://data.kitware.com/api/v1/item/5ff8b8d22fa25629b9d0e4ed/download
unzip -o ${DOWNLOAD_LOCATION}/download2.zip -d ${VIAME_INSTALL}

# PengHead -
wget -O ${DOWNLOAD_LOCATION}/download3.zip https://data.kitware.com/api/v1/item/6011ebf72fa25629b91aef03/download
unzip -o ${DOWNLOAD_LOCATION}/download3.zip -d ${VIAME_INSTALL}

# Ensure Download Location is Removed
rm -rf ${DOWNLOAD_LOCATION}
