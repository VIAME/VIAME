VIAME Network Installer Creation
================================

This directory contains the tools and templates for creating the VIAME
network installer, which allows users to selectively install VIAME components.

Prerequisites
-------------

- WiX Toolset 3.11+ (https://wixtoolset.org/)
- Python 3.8+ with ``girder-client`` package
- CMake 3.16+
- The staged VIAME MSI packages from the build process

File Overview
-------------

- ``installer_config.json`` - Configuration for versions, components, and model data
- ``generate_viame_installers.py`` - Python script to generate supplemental installers
- ``VIAME_Network.wxs`` - WiX bundle definition for the network installer
- ``VIAME_options.xml`` - WiX theme file defining the installer UI
- ``VIAME_32px.png`` - Logo for the installer

Step 1: Build VIAME Component Packages
--------------------------------------

Use the MSI build script to generate the 9 component packages::

    build_server_windows_msi.bat

This creates the following ZIP files in the build directory:

1. ``VIAME-Core.zip`` - Core (fletch + kwiver + vxl + opencv + python)
2. ``VIAME-CUDA.zip`` - CUDA/cuDNN support
3. ``VIAME-PyTorch.zip`` - PyTorch + pytorch-libs
4. ``VIAME-Extra-CPP.zip`` - Darknet, SVM, PostgreSQL
5. ``VIAME-DIVE.zip`` - DIVE GUI
6. ``VIAME-VIVIA.zip`` - VIVIA interface
7. ``VIAME-SEAL.zip`` - SEAL toolkit
8. ``VIAME-Models.zip`` - Pre-trained models
9. ``VIAME-Dev-Headers.zip`` - Development headers

Step 2: Convert ZIPs to MSI Packages
------------------------------------

Each ZIP file needs to be converted to an MSI package using CPack/WiX.
This is typically done by enabling ``VIAME_CREATE_INSTALLER`` in CMake
and building the ``PACKAGE`` target.

Copy the resultant ``.msi`` files to this directory.

Step 3: Generate Supplemental Model Installers
----------------------------------------------

The supplemental installers for model data are generated via the Python script.

Setting up credentials
++++++++++++++++++++++

Set the following environment variables::

    SET GIRDER_USER=your_username
    SET GIRDER_PASSWORD=your_password
    SET GIRDER_FOLDER_ID=parent_collection_id

Or use an API key::

    SET GIRDER_API_KEY=your_api_key
    SET GIRDER_FOLDER_ID=parent_collection_id

Updating the configuration
++++++++++++++++++++++++++

Edit ``installer_config.json`` to:

- Update version numbers in ``viame_version`` and ``data_version``
- Add/remove model data entries in ``model_data`` section
- Update external installer versions in ``external_installers``

Running the script
++++++++++++++++++

Execute the script::

    python generate_viame_installers.py

Options:

- ``--remake-all`` - Regenerate all installers even if they exist
- ``-j N`` - Use N parallel processes (default: 4)
- ``--config PATH`` - Use alternative config file
- ``-v`` - Verbose logging

This will:

1. Download external installers (e.g., VIAME-Dive)
2. Generate MSI packages for each model data entry
3. Upload installers to Girder
4. Create ``VIAME_Chain_File.wxs`` for the bundle

Step 4: Update Version Numbers
------------------------------

Before building the network installer, update the version variables in
``VIAME_Network.wxs``::

    <?define VIAMEVersion = "1.0.0" ?>
    <?define DiveVersion = "1.3.0" ?>

Step 5: Build the Network Installer
-----------------------------------

Build the WiX bundle using the WiX Toolset::

    candle.exe -ext WixBalExtension VIAME_Network.wxs
    light.exe -ext WixBalExtension VIAME_Network.wixobj

This produces ``VIAME_Network.exe``, the final network installer.

Installer UI Options
--------------------

The installer presents users with checkboxes to select components:

**Core Components:**

- CUDA/cuDNN Support
- PyTorch + Deep Learning Libraries
- Extra C++ Detectors (Darknet, SVM, PostgreSQL)

**GUI Interfaces:**

- DIVE GUI (Web-based annotation)
- VIVIA Interface (Desktop Qt/VTK application)

**Advanced Options:**

- SEAL Toolkit
- Pre-trained Models
- Development Headers

**Model Data Packages:**

- Sea Lion, MOUSS, SEFSC, Arctic Seals, HabCam models

Troubleshooting
---------------

**Girder authentication fails:**
  Ensure environment variables are set correctly. Try using an API key
  instead of username/password.

**MSI generation fails:**
  Check that WiX Toolset is installed and ``candle.exe``/``light.exe``
  are in the PATH.

**Download fails:**
  Check network connectivity and verify Girder item IDs are correct
  in the configuration file.

**Package exceeds 2GB:**
  MSI has a 2GB limit. Use the staged build approach to split large
  components into separate packages.
