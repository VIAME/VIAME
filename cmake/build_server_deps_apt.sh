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
gcc-12 \
g++-12 \
gfortran \
zlib1g-dev \
bzip2 \
libbz2-dev \
liblzma-dev

# Install python system packages
apt-get install -y python3-dev \
python3-pip \
python-is-python3

# Install python pip packages
python -m pip install numpy==1.25.2
