#!/bin/sh

# Configurable Input Paths
export VIAME_INSTALL="$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)/.."
export DOWNLOAD_LOCATION=/tmp/VIAME-Addons

# Ensure Download Location is Created
mkdir -p ${DOWNLOAD_LOCATION}

# Download All Optional Packages

# Habcam -
wget -O ${DOWNLOAD_LOCATION}/download1.zip https://viame.kitware.com/api/v1/item/613ed525a28c6cb53f3a0601/download
unzip -o ${DOWNLOAD_LOCATION}/download1.zip -d ${VIAME_INSTALL}

# SEFSC -
wget -O ${DOWNLOAD_LOCATION}/download2.zip https://viame.kitware.com/api/v1/item/6252405656394432c3c225e4/download
unzip -o ${DOWNLOAD_LOCATION}/download2.zip -d ${VIAME_INSTALL}

# PengHead -
wget -O ${DOWNLOAD_LOCATION}/download3.zip https://data.kitware.com/api/v1/item/6011ebf72fa25629b91aef03/download
unzip -o ${DOWNLOAD_LOCATION}/download3.zip -d ${VIAME_INSTALL}

# Motion -
wget -O ${DOWNLOAD_LOCATION}/download4.zip https://data.kitware.com/api/v1/item/601b00d02fa25629b9391ad6/download
unzip -o ${DOWNLOAD_LOCATION}/download4.zip -d ${VIAME_INSTALL}

# EM Tuna -
wget -O ${DOWNLOAD_LOCATION}/download5.zip https://data.kitware.com/api/v1/item/601afdde2fa25629b9390c41/download
unzip -o ${DOWNLOAD_LOCATION}/download5.zip -d ${VIAME_INSTALL}

# MOUSS Deep 7 -
wget -O ${DOWNLOAD_LOCATION}/download6.zip https://viame.kitware.com/api/v1/item/616d9bf7b53494023680648c/download
unzip -o ${DOWNLOAD_LOCATION}/download6.zip -d ${VIAME_INSTALL}

# Aerial Penguin
wget -O ${DOWNLOAD_LOCATION}/download7.zip https://viame.kitware.com/api/v1/item/615bc7aa7e5c13a5bb9af7a7/download
unzip -o ${DOWNLOAD_LOCATION}/download7.zip -d ${VIAME_INSTALL}

# Sea Lion
wget -O ${DOWNLOAD_LOCATION}/download8.zip https://viame.kitware.com/api/v1/item/61d341fa2a5486eb4d8d7933/download
unzip -o ${DOWNLOAD_LOCATION}/download8.zip -d ${VIAME_INSTALL}

# Ensure Download Location is Removed
rm -rf ${DOWNLOAD_LOCATION}
