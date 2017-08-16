#!/bin/sh
set -e

INSTALL_DIR=$HOME/deps
FLETCH_DIR=/opt/fletch
export PATH=$INSTALL_DIR/bin:$FLETCH_DIR/bin:$PATH
HASH_DIR=$FLETCH_DIR/hashes
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
TAR_FILE_ID=59822a8e8d777f16d01ea140
wget https://data.kitware.com/api/v1/file/$TAR_FILE_ID/hashsum_file/sha512 -O fletch.sha512
RHASH=`cat fletch.sha512`
echo "Current Fletch tarball hash: " $RHASH
if [ -f $HASH_FILE ] && [ -n "$RHASH" ] && grep -q $RHASH $HASH_FILE ; then
  echo "Using cached Fletch download"
else
  wget https://data.kitware.com/api/v1/file/$TAR_FILE_ID/download -O fletch.tgz
  rm -rf $FLETCH_DIR/*
  tar -xzf fletch.tgz -C $FLETCH_DIR
  cp fletch.sha512 $HASH_FILE
fi
