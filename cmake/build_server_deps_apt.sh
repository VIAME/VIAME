#! /bin/bash

# Install Fletch and VIAME system deps
apt-get update -y

apt-get install -y zip \
git \
wget \
tar \
libgl1-mesa-dev \
libexpat1-dev \
libgtk2.0-dev \
libxt-dev \
libxml2-dev \
liblapack-dev \
openssl \
libssl-dev \
curl \
libcurl4-openssl-dev \
g++ \
gfortran \
zlib1g-dev \
bzip2 \
libbz2-dev \
liblzma-dev \
python3-dev \
python3-pip \
python-is-python3

python -m pip install numpy==1.25.2
