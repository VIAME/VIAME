#! /bin/bash

# Install Fletch & VIAME system deps
yum -y groupinstall 'Development Tools'
yum install -y zip \
git \
wget \
zlib \
zlib-devel \
freeglut-devel \
freetype-devel \
mesa-libGLU-devel \
lapack-devel \
libffi-devel \
libXt-devel \
libXmu-devel \
libXi-devel \
expat-devel \
readline-devel \
curl \
curl-devel \
atlas-devel \
file \
which \
bzip2 \
bzip2-devel \
xz-devel \
vim \
devtoolset-9 \
perl-IPC-Cmd
