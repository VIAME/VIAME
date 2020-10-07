#!/bin/sh
set -e

INSTALL_DIR=$HOME/deps
FLETCH_DIR=/opt/kitware/fletch
export PATH=$INSTALL_DIR/bin:$FLETCH_DIR/bin:$PATH
HASH_DIR=/opt/kitware/hashes
mkdir -p $FLETCH_DIR
mkdir -p $HASH_DIR
mkdir -p $INSTALL_DIR

# Make a directory to test installation of KWIVER into
mkdir -p $HOME/install

# download and unpack Fletch
HASH_FILE="$HASH_DIR/fletch.sha512"
cd /tmp
if [ -f $TRAVIS_BUILD_DIR/doc/release-notes/master.txt ]; then
  TAR_FILE_ID=5d3a2d40877dfcc9022ec9f5
  echo "Using master branch of Fletch"
else
  TAR_FILE_ID=5d3f0c94877dfcc90235f064
  echo "Using release branch of Fletch"
fi

wget https://data.kitware.com/api/v1/file/$TAR_FILE_ID/hashsum_file/sha512 -O fletch.sha512
RHASH=`cat fletch.sha512`
echo "Current Fletch tarball hash: " $RHASH
if [ -f $HASH_FILE ] && [ -n "$RHASH" ] && grep -q $RHASH $HASH_FILE ; then
  echo "Using cached Fletch download"
else
  wget https://data.kitware.com/api/v1/file/$TAR_FILE_ID/download -O fletch.tgz
  rm -rf $FLETCH_DIR/*
  tar -xzf fletch.tgz -C /opt/kitware
  cp fletch.sha512 $HASH_FILE
fi

# Make a directory for cmake binaries
mkdir -p $HOME/cmake_install
cd $HOME/cmake_install
wget https://cmake.org/files/v3.15/cmake-3.15.7-Linux-x86_64.sh
chmod +x cmake-3.15.7-Linux-x86_64.sh
yes | ./cmake-3.15.7-Linux-x86_64.sh
