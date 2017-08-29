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

# check if directory is cached
if [ ! -f "$INSTALL_DIR/bin/cmake" ]; then
  cd /tmp
  wget --no-check-certificate https://cmake.org/files/v3.4/cmake-3.4.0-Linux-x86_64.sh
  bash cmake-3.4.0-Linux-x86_64.sh --skip-license --prefix="$INSTALL_DIR/"
else
  echo 'Using cached CMake directory.';
fi


# download and unpack Fletch
HASH_FILE="$HASH_DIR/fletch.sha512"
cd /tmp
if [ -f $TRAVIS_BUILD_DIR/doc/release-notes/master.txt ]; then
  TAR_FILE_ID=599c39468d777f7d33e9cbe5
  echo "Using master branch of Fletch"
else
  TAR_FILE_ID=599f2db18d777f7d33e9cc9e
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
