#!/bin/sh

# Configurable Input Paths
export VIAME_LOCATION="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/.."
export DOWNLOAD_LOCATION=~/VIAME-Addons

# Ensure Download Location is Created
mkdir -p ${DOWNLOAD_LOCATION}

# Download All Optional Packages
wget -O ${DOWNLOAD_LOCATION}/download1.zip https://data.kitware.com/api/v1/item/5f6bb7e850a41e3d19a63047/download
tar -xvf ${DOWNLOAD_LOCATION}/download1.zip -C ${VIAME_INSTALL}

# Ensure Download Location is Removed
rm -rf ${DOWNLOAD_LOCATION}
