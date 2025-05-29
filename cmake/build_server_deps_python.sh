#! /bin/bash

# Install python system packages
apt-get install -y python3-dev \
python3-pip \
python-is-python3

# Install python pip packages
python -m pip install numpy==1.25.2
