VIAME Network Installer Creation
================================

This directory contains the tools and templates for creating the VIAME
network installer, which allows users to selectively install VIAME components.

Prerequisites
-------------

- WiX Toolset v5.0+ (https://wixtoolset.org/)

  Install via .NET tool::

      dotnet tool install --global wix
      wix extension add WixToolset.Bal.wixext

- Python 3.8+ with ``girder-client`` package
- CMake 3.16+
- The staged VIAME MSI packages from the build process

File Overview
-------------

- ``msi_installer_config.json`` - Configuration for versions and components
- ``download_viame_addons.csv`` - Model data definitions (name, URL, description, MD5, dependencies)
- ``msi_generate_installer.py`` - Python script to generate UI and supplemental installers
- ``msi_viame_network.wxs`` - WiX v5 bundle definition for the network installer
- ``msi_viame_options.xml`` - WiX v5 theme file (auto-generated from CSV)
- ``msi_viame_chain_file.wxs`` - Model packages chain file (auto-generated)
- ``viame-icon-32px.png`` - Logo for the installer

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

Edit ``msi_installer_config.json`` to:

- Update version numbers in ``viame_version`` and ``data_version``
- Update external installer versions in ``external_installers``

Edit ``download_viame_addons.csv`` to add/remove model packages:

- Format: ``name, download_url, description, md5sum, {DEP1, DEP2}``
- Dependencies can be: PYTORCH, PYTORCH-NETHARN, CUDA

Running the script
++++++++++++++++++

Generate just the UI (no Girder authentication required)::

    python msi_generate_installer.py --ui-only

Full generation with model installers::

    python msi_generate_installer.py

Options:

- ``--ui-only`` - Only regenerate the UI theme file (no Girder needed)
- ``--remake-all`` - Regenerate all installers even if they exist
- ``-j N`` - Use N parallel processes (default: 4)
- ``--config PATH`` - Use alternative config file
- ``--addons-csv PATH`` - Use alternative model addons CSV file
- ``-v`` - Verbose logging

This will:

1. Parse model data from ``download_viame_addons.csv``
2. Generate ``msi_viame_options.xml`` with model checkboxes
3. Download external installers (e.g., VIAME-Dive)
4. Generate MSI packages for each model data entry
5. Upload installers to Girder
6. Create ``msi_viame_chain_file.wxs`` for the bundle

Step 4: Update Version Numbers
------------------------------

Before building the network installer, update the version variables in
``msi_viame_network.wxs``::

    <?define VIAMEVersion = "1.0.0" ?>
    <?define DiveVersion = "1.3.0" ?>

Step 5: Build the Network Installer
-----------------------------------

Build the WiX v5 bundle using the ``wix`` CLI tool::

    wix build -ext WixToolset.Bal.wixext msi_viame_network.wxs -o viame-installer.exe

Or with additional extensions if needed::

    wix build -ext WixToolset.Bal.wixext -ext WixToolset.Util.wixext msi_viame_network.wxs -o viame-installer.exe

This produces ``viame-installer.exe``, the final network installer.

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

Model packages are auto-generated from ``download_viame_addons.csv`` and displayed
as checkboxes in the installer. Each model's dependencies (e.g., PyTorch) are
automatically enforced - if a model requires PyTorch, the user must also select
the PyTorch component.

Troubleshooting
---------------

**Girder authentication fails:**
  Ensure environment variables are set correctly. Try using an API key
  instead of username/password. Use ``--ui-only`` to test UI generation
  without Girder.

**WiX build fails:**
  Ensure WiX v5 is installed via ``dotnet tool install --global wix``.
  Check that the Bal extension is added: ``wix extension add WixToolset.Bal.wixext``.

**Download fails:**
  Check network connectivity and verify URLs in ``download_viame_addons.csv``
  are accessible.

**Package exceeds 2GB:**
  MSI has a 2GB limit. Use the staged build approach to split large
  components into separate packages.

**Theme file changes not appearing:**
  Regenerate the theme with ``python msi_generate_installer.py --ui-only``
  after modifying ``download_viame_addons.csv``.
