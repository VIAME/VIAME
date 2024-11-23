#! /bin/bash

# Fletch, VIAME, CMAKE system deps
wget https://ftp.openssl.org/source/openssl-1.1.1k.tar.gz
tar -xzvf openssl-1.1.1k.tar.gz
cd openssl-1.1.1k
./config --prefix=/usr --openssldir=/etc/ssl --libdir=lib no-shared zlib-dynamic
make -j$(nproc)
make install
cd /
rm -rf openssl-1.1.1k.tar.gz openssl-1.1.1k
