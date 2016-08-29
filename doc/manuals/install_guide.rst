=================
Installing Kwiver
=================

These instructions are designed to help build Kwiver on a fresh machine. They were written for and tested on Ubuntu 16.04 Desktop version. Other Linux machines will have similar directions, but some steps (particularly the dependency install) may not be totally identical.

********************
Install Dependencies
********************

Some of the dependencies required for Kwiver can be installed with one quick and easy instruction with no configuration required. Different Linux distributions may have different packages already installed, or may use a different package manager than apt, but even on Ubuntu this should help to provide a starting point.

`sudo apt-get install git zlib1g-dev libcurl4-openssl-dev libexpat1-dev dh-autoreconf liblapack-dev libxt-dev`

`sudo apt-get build-dep libboost-all-dev qt5-default`

Install CMAKE
=============

The version of cmake you currently get with apt is too old to use for kwiver, so you need to do a manual install. Go to the cmake website, `https://cmake.org/download`, and download the appropriate binary distribution (for Ubuntu, this would be something like cmake-3.6.1-Linux-x86_64.sh, depending on version). Download the source code, cmake-3.6.1.tar.gz (or just download and use the installer for windows).  To untar and build the source, use the following set of commands. Keep in mind that if you're not using version 3.6.1, you'll need to update the version number to match your download.

`cd ~/Downloads`

`tar zxfv cmake-3.6.1.tar.gz`

`cd cmake-3.6.1`

`./bootstrap --system-curl --no-system-libs`

`make`

`sudo make install`

`sudo ln -s /usr/local/bin/cmake /bin/cmake`

These instructions build the source code into a working executable, installs the executable into a personal directory, and then lets the operating system know where that directory is so it can find cmake in the future.

**************
Install Fletch
**************

Fletch is a CMake driven build that will help configure and install a series of component packages necessary for Kwiver, like Eigen and Boost. Navigate to the directory where you want to put your source code and builds. I personally like to use ~/Work and then set up a new directory for each repo. With all dependencies for Fletch installed in the last couple of steps, Fletch should build without any issues.

`mkdir fletch`

`cd fletch`

`git clone https://github.com/kitware/fletch.git`

`mkdir build`

`cd build`

`cmake -Dfletch_ENABLE_ALL_PACKAGES:bool=on ../fletch`

`make`

**************
Install Kwiver
**************

After Fletch is built, you should have everything necessary to build Kwiver. Navigate back to the directory you want to put Kwiver in (if you followed the directions above, the command to return is `cd ../..`). In the cmake step, make sure to fill in your Fletch build directory so Kwiver knows where to find its dependencies. For example, I would use `cmake -Dfletch_DIR:path=/home/dave/Work/fletch/build ../kwiver`.

`mkdir kwiver`

`cd kwiver`

`git clone https://github.com/kitware/kwiver.git`

`mkdir build`

`cd build`

`cmake -Dfletch_DIR:path=<fletch_build_directory> ../kwiver`

`make`