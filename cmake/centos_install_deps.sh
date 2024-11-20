#! /bin/bash

# Updates for old centos yum external repos
../cmake/centos_fix_yum_links.sh
yum -y install centos-release-scl
../cmake/centos_fix_yum_links.sh

# Install Fletch & VIAME system deps
yum -y groupinstall 'Development Tools'
yum install -y zip \
git \
wget \
openssl \
openssl-devel \
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
devtoolset-7

# Install NINJA for faster builds of some dependencies
rpm -ivh https://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/n/ninja-build-1.10.2-3.el7.x86_64.rpm

