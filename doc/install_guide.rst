================
Installing Viame
================

These instructions are designed to help build Viame on a fresh machine. They were written for and tested on Ubuntu 16.04 Desktop version. Other Linux machines will have similar directions, but some steps (particularly the dependency install) may not be totally identical.

********************
Install Dependencies
********************

Some of the dependencies required for Viame can be installed with one quick and easy instruction with no configuration required. Different Linux distributions may have different packages already installed, or may use a different package manager than apt, but even on Ubuntu this should help to provide a starting point.

`sudo apt-get install git zlib1g-dev libcurl4-openssl-dev libexpat1-dev dh-autoreconf liblapack-dev libxt-dev`

`sudo apt-get build-dep libboost-all-dev qt5-default`

Install CMAKE
=============

The version of cmake you currently get with apt is too old to use for Viame, so you need to do a manual install. Go to the cmake website, `https://cmake.org/download`, and download the appropriate binary distribution (for Ubuntu, this would be something like cmake-3.6.1-Linux-x86_64.sh, depending on version). Download the source code, cmake-3.6.1.tar.gz (or just download and use the installer for windows).  To untar and build the source, use the following set of commands. Keep in mind that if you're not using version 3.6.1, you'll need to update the version number to match your download.

`cd ~/Downloads`

`tar zxfv cmake-3.6.1.tar.gz`

`cd cmake-3.6.1`

`./bootstrap --system-curl --no-system-libs`

`make`

`sudo make install`

`sudo ln -s /usr/local/bin/cmake /bin/cmake`

These instructions build the source code into a working executable, installs the executable into a personal directory, and then lets the operating system know where that directory is so it can find cmake in the future.

