#!/bin/bash

python3 /home/provision/init_girder.py
girder serve --database $MONGO_URI --host 0.0.0.0
