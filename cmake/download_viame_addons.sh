#!/bin/sh

# Configurable Input Paths
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/.."
export DOWNLOAD_LOCATION=/tmp/VIAME-Addons

# Ensure Download Location is Created
mkdir -p ${DOWNLOAD_LOCATION}

# Download All Optional Packages

# Habcam -
wget -O ${DOWNLOAD_LOCATION}/download1.zip https://viame.kitware.com/api/v1/item/627b145487bad2e19a4c4697/download
unzip -o ${DOWNLOAD_LOCATION}/download1.zip -d ${VIAME_INSTALL}

# SEFSC -
wget -O ${DOWNLOAD_LOCATION}/download2.zip https://viame.kitware.com/api/v1/item/627b32b1994809b024f207a7/download
unzip -o ${DOWNLOAD_LOCATION}/download2.zip -d ${VIAME_INSTALL}

# PengHead -
wget -O ${DOWNLOAD_LOCATION}/download3.zip https://viame.kitware.com/api/v1/item/627b3289ea630db5587b577d/download
unzip -o ${DOWNLOAD_LOCATION}/download3.zip -d ${VIAME_INSTALL}

# Motion -
wget -O ${DOWNLOAD_LOCATION}/download4.zip https://viame.kitware.com/api/v1/item/627b326fea630db5587b577b/download
unzip -o ${DOWNLOAD_LOCATION}/download4.zip -d ${VIAME_INSTALL}

# EM Tuna -
wget -O ${DOWNLOAD_LOCATION}/download5.zip https://viame.kitware.com/api/v1/item/627b326cc4da86e2cd3abb5b/download
unzip -o ${DOWNLOAD_LOCATION}/download5.zip -d ${VIAME_INSTALL}

# MOUSS Deep 7 -
wget -O ${DOWNLOAD_LOCATION}/download6.zip https://viame.kitware.com/api/v1/item/627b3282c4da86e2cd3abb5d/download
unzip -o ${DOWNLOAD_LOCATION}/download6.zip -d ${VIAME_INSTALL}

# Aerial Penguin
wget -O ${DOWNLOAD_LOCATION}/download7.zip https://viame.kitware.com/api/v1/item/615bc7aa7e5c13a5bb9af7a7/download
unzip -o ${DOWNLOAD_LOCATION}/download7.zip -d ${VIAME_INSTALL}

# Sea Lion
wget -O ${DOWNLOAD_LOCATION}/download8.zip https://viame.kitware.com/girder/api/v1/item/627b0b877b5df7aa226545ef/download
unzip -o ${DOWNLOAD_LOCATION}/download8.zip -d ${VIAME_INSTALL}

# Ensure Download Location is Removed
rm -rf ${DOWNLOAD_LOCATION}
