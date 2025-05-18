#! /bin/bash

# Install Fletch and VIAME system deps
yum -y update

yum -y groupinstall 'Development Tools'

yum install -y zip \
git \
wget \
zlib \
zlib-devel \
freeglut-devel \
freetype-devel \
mesa-libGLU-devel \
libffi-devel \
libXt-devel \
libXmu-devel \
libXi-devel \
expat-devel \
readline-devel \
openssl \
openssl-devel \
curl-devel \
atlas-devel \
file \
which \
bzip2 \
bzip2-devel \
xz-devel \
vim \
perl-IPC-Cmd
