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

******************
Prepare the Source
******************

With all our dependencies installed, we need to build the environment for Viame itself. Viame uses git submodules rather than requiring the user to grab each repository totally separately. To prepare the environment and obtain all the necessary source code, use the following commands. Note that you can change `src` to whatever you want to name your Viame source directory.

`git clone git@github.com:Kitware/VIAME git src`

`cd src`

`git submodule init`

`git submodule update`

***********
Build Viame
***********

Viame may be built with a number of optional plugins--VXL, Caffe, OpenCV, Scallop_TK, and Matlab--with a corresponding option called VIAME_ENABLE_[option], in all caps. For each plugin to install, you need a cmake build flag setting the option. The flag looks like `-DVIAME_ENABLE_OPENCV:BOOL=ON`, of course changing OPENCV to match the plugin. Multiple plugins may be used, or none.

Viame is meant to be built and installed in the same directory as the source code, so stay in the src directory and run the following commands:

`mkdir build`

`cd build`

`cmake [build_flags] ..` 

`make`

