# Kitware 2018:
# Setup and run script for KWIVER docker 1.2.0
#
# Function:
# - create a docker image that has Fletch and Kwiver built named kwiver:1.2.0
# - create a docker container named kwiver1.2.0
#
# Configuration:
# - Script must run in /src/docker/ directory for pathing purposes
# - Script optionally sets up a mounted shared volume between host/docker

# Initial building of image
docker build --force-rm -t kwiver:1.2.0 .

# Starting container without shared volume (default)
docker run -d --name kwiver1.2.0 -it kwiver:1.2.0 /bin/bash

#	OR

# Starting container with shared volume
# **This will create a shared folder at /kwiver/src/docker/SharedKIWVER/KWIVER1.2.0 on host and /SharedKWIVER/KWIVER1.2.0 in container
#docker run -d --name kwiver1.2.0 -it -v $PWD/SharedKWIVER/KWIVER1.2.0:/SharedKWIVER/KWIVER1.2.0 kwiver:1.2.0 /bin/bash

