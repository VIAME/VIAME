#! /bin/bash

# Download opencv files locally instead of from notoriously often failing opencv repo
curl https://data.kitware.com/api/v1/item/682bf0110dcd2dfb445a5404/download --output tmp.tar.gz
tar -xvf tmp.tar.gz
rm tmp.tar.gz
