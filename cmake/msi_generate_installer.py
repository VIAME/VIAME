#!/usr/bin/env python3
"""
VIAME Network Installer Generator

Generates MSI installers for VIAME components and model data packages,
uploads them to Girder, and creates WiX chain files for the network installer.

Model information is read from download_viame_addons.csv which contains:
  - Model name
  - Download URL (.zip)
  - Description
  - MD5 checksum
  - Dependencies (e.g., {PYTORCH, PYTORCH-NETHARN})

Usage:
    python generate_viame_installers.py [options]

Environment Variables:
    GIRDER_USER     - Girder username for authentication
    GIRDER_PASSWORD - Girder password for authentication
    GIRDER_API_KEY  - Alternative: Girder API key (instead of user/password)
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing
import os
import re
import shutil
import signal
import sys
import urllib.request
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import run, CalledProcessError
from tempfile import TemporaryDirectory
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directory containing this script
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "msi_installer_config.json"
ADDONS_CSV_FILE = SCRIPT_DIR / "download_viame_addons.csv"


@dataclass
class ModelAddon:
    """Represents a model addon from the CSV file."""
    name: str
    download_url: str
    description: str
    md5sum: str
    dependencies: list[str] = field(default_factory=list)

    @property
    def safe_name(self) -> str:
        """Get a filesystem/identifier-safe name."""
        # Remove spaces and special characters, convert to PascalCase
        return re.sub(r'[^a-zA-Z0-9]', '', self.name.title().replace(' ', ''))

    @property
    def checkbox_name(self) -> str:
        """Get the checkbox name for WiX UI."""
        return f"{self.safe_name}ModelCheckbox"


def parse_addons_csv(csv_path: Path = ADDONS_CSV_FILE) -> list[ModelAddon]:
    """
    Parse the download_viame_addons.csv file.

    CSV format:
      name, download_url, description, md5sum, {DEP1, DEP2}

    Returns:
        List of ModelAddon objects
    """
    if not csv_path.exists():
        logger.warning(f"Addons CSV not found: {csv_path}")
        return []

    models = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse the CSV line - handle the dependencies in braces
            # Format: name, url, description, md5, {dep1, dep2}
            parts = []
            current = []
            in_braces = False

            for char in line:
                if char == '{':
                    in_braces = True
                    current.append(char)
                elif char == '}':
                    in_braces = False
                    current.append(char)
                elif char == ',' and not in_braces:
                    parts.append(''.join(current).strip())
                    current = []
                else:
                    current.append(char)
            parts.append(''.join(current).strip())

            if len(parts) < 4:
                logger.warning(f"Skipping malformed line {line_num}: {line[:50]}...")
                continue

            name = parts[0].strip()
            download_url = parts[1].strip()
            description = parts[2].strip()
            md5sum = parts[3].strip()

            # Parse dependencies from {DEP1, DEP2} format
            dependencies = []
            if len(parts) >= 5:
                deps_str = parts[4].strip()
                if deps_str.startswith('{') and deps_str.endswith('}'):
                    deps_inner = deps_str[1:-1]
                    dependencies = [d.strip() for d in deps_inner.split(',') if d.strip()]

            if name and download_url:
                models.append(ModelAddon(
                    name=name,
                    download_url=download_url,
                    description=description,
                    md5sum=md5sum,
                    dependencies=dependencies
                ))

    logger.info(f"Parsed {len(models)} model addons from CSV")
    return models


@dataclass
class InstallerConfig:
    """Configuration for the installer generator."""
    viame_version: dict[str, int]
    data_version: dict[str, int]
    girder_api_url: str
    girder_data_url: str
    components: dict[str, dict]
    model_addons: list[ModelAddon]
    external_installers: dict[str, dict]
    num_processes: int
    girder_folder_id: str | None = None

    @classmethod
    def load(cls, config_path: Path = CONFIG_FILE,
             addons_csv_path: Path = ADDONS_CSV_FILE) -> 'InstallerConfig':
        """Load configuration from JSON file and model addons from CSV."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            data = json.load(f)

        # Parse model addons from CSV instead of JSON config
        model_addons = parse_addons_csv(addons_csv_path)

        return cls(
            viame_version=data['viame_version'],
            data_version=data['data_version'],
            girder_api_url=data['girder']['api_url'],
            girder_data_url=data['girder']['data_url'],
            components=data.get('components', {}),
            model_addons=model_addons,
            external_installers=data.get('external_installers', {}),
            num_processes=data.get('build_settings', {}).get('num_processes', 4),
            girder_folder_id=os.environ.get('GIRDER_FOLDER_ID')
        )

    @property
    def viame_version_string(self) -> str:
        """Get VIAME version as string."""
        v = self.viame_version
        return f"{v['major']}.{v['minor']}.{v['patch']}"

    @property
    def data_version_string(self) -> str:
        """Get data version as string."""
        v = self.data_version
        return f"{v['major']}.{v['minor']}"


# CMake template for model data package installers
# Downloads a zip and extracts contents to the VIAME install directory
CMAKE_TEMPLATE = """
cmake_minimum_required(VERSION 3.16)
project({data_name}Data NONE)

set(ZIP_FILE "${{CMAKE_BINARY_DIR}}/{data_name}.zip")
set(EXTRACT_DIR "${{CMAKE_BINARY_DIR}}/extracted")

# Download the model zip
if(NOT EXISTS ${{ZIP_FILE}})
  message(STATUS "Downloading {data_name} model...")
  file(DOWNLOAD {download_url}
    ${{ZIP_FILE}}
    SHOW_PROGRESS
    STATUS download_status
    EXPECTED_MD5 {md5sum})
  list(GET download_status 0 status_code)
  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "Download failed: ${{download_status}}")
  endif()
endif()

# Extract the zip contents
if(NOT EXISTS ${{EXTRACT_DIR}})
  message(STATUS "Extracting {data_name} model...")
  file(MAKE_DIRECTORY ${{EXTRACT_DIR}})
  execute_process(
    COMMAND ${{CMAKE_COMMAND}} -E tar xf ${{ZIP_FILE}}
    WORKING_DIRECTORY ${{EXTRACT_DIR}}
    RESULT_VARIABLE extract_result)
  if(NOT extract_result EQUAL 0)
    message(FATAL_ERROR "Extraction failed")
  endif()
endif()

# Install extracted contents to root (will merge with VIAME install)
install(DIRECTORY ${{EXTRACT_DIR}}/
  DESTINATION .)

set(CPACK_BINARY_WIX ON CACHE BOOL "Use WIX" FORCE)
set(CPACK_BINARY_NSIS OFF CACHE BOOL "No NSIS" FORCE)
set(CPACK_PACKAGE_NAME "{data_name}Data")
set(CPACK_PACKAGE_VERSION_MAJOR "{version_major}")
set(CPACK_PACKAGE_VERSION_MINOR "{version_minor}")
set(CPACK_PACKAGE_VERSION_PATCH "")
include(CPack)
"""

# WiX package template for chain file
WIX_PACKAGE_TEMPLATE = """<MsiPackage Id="{package_id}"
            Name="{file_name}"
            Compressed="no"
            DisplayInternalUI="yes"
            DownloadUrl="{download_url}"
            DisplayName="Downloading {display_name}"
            InstallCondition="{condition}">
</MsiPackage>
"""


class GirderClient:
    """Simple Girder client wrapper with authentication."""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        self._client = None

    def authenticate(self) -> None:
        """Authenticate with Girder using environment variables."""
        try:
            import girder_client
        except ImportError:
            raise ImportError(
                "girder_client not installed. Install with: pip install girder-client"
            )

        self._client = girder_client.GirderClient(apiUrl=self.api_url)

        # Try API key first, then username/password
        api_key = os.environ.get('GIRDER_API_KEY')
        if api_key:
            logger.info("Authenticating with Girder API key")
            self._client.authenticate(apiKey=api_key)
        else:
            username = os.environ.get('GIRDER_USER')
            password = os.environ.get('GIRDER_PASSWORD')
            if not username or not password:
                raise EnvironmentError(
                    "Girder credentials not found. Set either:\n"
                    "  - GIRDER_API_KEY environment variable, or\n"
                    "  - GIRDER_USER and GIRDER_PASSWORD environment variables"
                )
            logger.info(f"Authenticating with Girder as user: {username}")
            self._client.authenticate(username, password)

    @property
    def client(self):
        """Get the underlying girder_client instance."""
        if self._client is None:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        return self._client

    def get_or_create_folder(self, parent_id: str, name: str,
                             parent_type: str = 'collection') -> str:
        """Get or create a folder, returning its ID."""
        folder = self.client.createFolder(
            parent_id, name,
            parentType=parent_type,
            reuseExisting=True
        )
        return folder['_id']

    def find_item(self, folder_id: str, name: str) -> dict | None:
        """Find an item by name in a folder."""
        items = self.client.get("item", {"folderId": folder_id, "name": name})
        return items[0] if items else None

    def upload_file(self, folder_id: str, file_path: Path) -> dict:
        """Upload a file to a Girder folder."""
        file_size = file_path.stat().st_size
        with open(file_path, 'rb') as f:
            return self.client.uploadFile(
                folder_id, f, file_path.name, file_size,
                parentType='folder'
            )


class InstallerGenerator:
    """Generates MSI installers for VIAME components."""

    def __init__(self, config: InstallerConfig, options: dict[str, Any]):
        self.config = config
        self.options = options
        self.girder: GirderClient | None = None
        self.upload_folder_id: str | None = None

    def initialize(self) -> None:
        """Initialize Girder connection and upload folder."""
        self.girder = GirderClient(self.config.girder_api_url)
        self.girder.authenticate()

        if self.config.girder_folder_id:
            self.upload_folder_id = self.girder.get_or_create_folder(
                self.config.girder_folder_id,
                "Supplemental Installers",
                parent_type='collection'
            )
            logger.info(f"Upload folder ID: {self.upload_folder_id}")

    def generate_model_installer(self, model: ModelAddon, results: dict) -> None:
        """Generate an MSI installer for a model data package."""
        version_major = self.config.data_version['major']
        version_minor = self.config.data_version['minor']
        safe_name = model.safe_name
        target_name = f"{safe_name}Data-{version_major}.{version_minor}-win32.msi"

        # Check if already exists
        if self.upload_folder_id and not self.options.get('remake_all', False):
            existing = self.girder.find_item(self.upload_folder_id, target_name)
            if existing:
                logger.info(f"Skipping {model.name} - already exists")
                existing['itemId'] = existing['_id']
                existing['model'] = model
                results[target_name] = existing
                return

        logger.info(f"Generating installer for {model.name}...")

        try:
            with TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Write CMakeLists.txt
                cmake_content = CMAKE_TEMPLATE.format(
                    data_name=safe_name,
                    download_url=model.download_url,
                    md5sum=model.md5sum,
                    version_major=version_major,
                    version_minor=version_minor
                )
                (tmpdir_path / "CMakeLists.txt").write_text(cmake_content)

                # Run CMake configure and build
                run(["cmake", "."], cwd=tmpdir, check=True, capture_output=True)
                run(["cmake", "--build", ".", "--target", "PACKAGE"],
                    cwd=tmpdir, check=True, capture_output=True)

                # Find generated MSI
                msi_files = list(tmpdir_path.glob("*.msi"))
                if not msi_files:
                    raise RuntimeError(f"No MSI file generated for {model.name}")

                installer_path = msi_files[0]
                logger.info(f"Generated: {installer_path.name} "
                           f"({installer_path.stat().st_size / 1024 / 1024:.1f} MB)")

                # Upload to Girder
                if self.upload_folder_id:
                    result = self.girder.upload_file(
                        self.upload_folder_id, installer_path
                    )
                    result['model'] = model
                    results[installer_path.name] = result
                    logger.info(f"Uploaded {installer_path.name} to Girder")

                # Copy to script directory
                dest_path = SCRIPT_DIR / installer_path.name
                shutil.copy2(installer_path, dest_path)
                logger.info(f"Copied to {dest_path}")

        except CalledProcessError as e:
            logger.error(f"Build failed for {model.name}: {e.stderr.decode() if e.stderr else e}")
            raise
        except Exception as e:
            logger.error(f"Error generating installer for {model.name}: {e}")
            raise

    def download_external_installer(self, name: str, info: dict) -> Path:
        """Download an external installer (e.g., VIAME-Dive)."""
        version = info['version']
        url = info['download_url'].format(version=version)
        filename = f"{info['name']}-{version}.msi"
        dest_path = SCRIPT_DIR / filename

        if dest_path.exists() and not self.options.get('remake_all', False):
            logger.info(f"External installer already exists: {filename}")
            return dest_path

        logger.info(f"Downloading {filename} from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                with open(dest_path, 'wb') as f:
                    shutil.copyfileobj(response, f)
            logger.info(f"Downloaded: {filename} "
                       f"({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return dest_path
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise

    def _map_dependency_to_condition(self, dep: str) -> str:
        """Map CSV dependencies to WiX checkbox conditions."""
        dep_map = {
            'PYTORCH': 'PyTorchCheckbox=1',
            'PYTORCH-NETHARN': 'PyTorchCheckbox=1',
            'CUDA': 'CUDACheckbox=1',
        }
        return dep_map.get(dep.upper(), '')

    def generate_chain_file(self, installer_results: dict) -> Path:
        """Generate the WiX chain file for all model installers."""
        chain_file = SCRIPT_DIR / "msi_viame_chain_file.wxs"

        logger.info(f"Generating chain file: {chain_file}")

        content = ["<Include>"]

        for installer_name, installer_data in installer_results.items():
            model: ModelAddon = installer_data.get('model')
            if not model:
                logger.warning(f"No model info for {installer_name}, skipping")
                continue

            package_id = f"{model.safe_name}Data"

            # Build install condition: checkbox AND dependencies
            conditions = [f"{model.checkbox_name}=1"]
            for dep in model.dependencies:
                dep_condition = self._map_dependency_to_condition(dep)
                if dep_condition and dep_condition not in conditions:
                    conditions.append(dep_condition)

            condition = " AND ".join(conditions)

            download_url = (
                f"{self.config.girder_api_url}/item/{installer_data['itemId']}/download"
            )

            content.append(WIX_PACKAGE_TEMPLATE.format(
                package_id=package_id,
                file_name=installer_name,
                download_url=download_url,
                display_name=f"{model.name} Model",
                condition=condition
            ))

        content.append("</Include>")

        chain_file.write_text("\n".join(content))
        logger.info(f"Chain file generated with {len(installer_results)} packages")

        return chain_file

    def generate_ui_fragment(self) -> Path:
        """Generate complete msi_viame_options.xml with model checkboxes from CSV."""
        ui_file = SCRIPT_DIR / "msi_viame_options.xml"

        logger.info(f"Generating options UI: {ui_file}")

        # Calculate layout for model checkboxes
        # Two columns, starting after the "Additional model data packages:" text
        model_start_y = 518
        row_height = 17
        col_width = 210
        x_col1 = 20
        x_col2 = 230

        # Calculate window height based on number of models
        num_rows = (len(self.config.model_addons) + 1) // 2
        model_section_height = num_rows * row_height + 20
        window_height = model_start_y + model_section_height + 50

        # Generate model checkbox XML
        model_checkboxes = []
        for i, model in enumerate(self.config.model_addons):
            row = i // 2
            col = i % 2
            x = x_col1 if col == 0 else x_col2
            y = model_start_y + (row * row_height)

            # Truncate label if too long for UI
            label = model.name
            if len(label) > 22:
                label = label[:19] + "..."

            model_checkboxes.append(
                f'        <Checkbox Name="{model.checkbox_name}" '
                f'X="{x}" Y="{y}" Width="{col_width - 10}" Height="15" '
                f'TabStop="yes" FontId="3">{label}</Checkbox>'
            )

        model_checkboxes_xml = '\n'.join(model_checkboxes)

        # Complete theme XML template
        theme_xml = f'''<?xml version="1.0" encoding="utf-8"?>
<!--
  VIAME Network Installer Options Theme

  This theme file defines the UI for the WiX bootstrapper application.
  Model checkboxes are auto-generated from download_viame_addons.csv.

  DO NOT EDIT MODEL CHECKBOXES MANUALLY - run msi_generate_installer.py to regenerate.
-->
<Theme xmlns="http://wixtoolset.org/schemas/thmutil/2010">
    <Window Width="520" Height="{window_height}" HexStyle="100a0000" FontId="0">#(loc.Caption)</Window>
    <Font Id="0" Height="-12" Weight="500" Foreground="000000" Background="FFFFFF">Segoe UI</Font>
    <Font Id="1" Height="-24" Weight="500" Foreground="000000">Segoe UI</Font>
    <Font Id="2" Height="-22" Weight="500" Foreground="666666">Segoe UI</Font>
    <Font Id="3" Height="-12" Weight="500" Foreground="000000" Background="FFFFFF">Segoe UI</Font>
    <Font Id="4" Height="-12" Weight="500" Foreground="ff0000" Background="FFFFFF" Underline="yes">Segoe UI</Font>
    <Font Id="5" Height="-11" Weight="400" Foreground="666666" Background="FFFFFF">Segoe UI</Font>

    <Image X="11" Y="11" Width="64" Height="64" ImageFile="logo.png" Visible="yes"/>
    <Text X="80" Y="11" Width="-11" Height="64" FontId="1" Visible="yes">#(loc.Title)</Text>

    <Page Name="Install">
        <Hypertext Name="EulaHyperlink" X="11" Y="121" Width="-11" Height="51" TabStop="yes" FontId="3" HideWhenDisabled="yes">#(loc.InstallLicenseLinkText)</Hypertext>
        <Checkbox Name="EulaAcceptCheckbox" X="-11" Y="-41" Width="246" Height="17" TabStop="yes" FontId="3" HideWhenDisabled="yes">#(loc.InstallAcceptCheckbox)</Checkbox>
        <Button Name="OptionsButton" X="-171" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0" HideWhenDisabled="yes">#(loc.InstallOptionsButton)</Button>
        <Button Name="InstallButton" X="-91" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0">#(loc.InstallInstallButton)</Button>
        <Button Name="WelcomeCancelButton" X="-11" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0">#(loc.InstallCloseButton)</Button>
    </Page>

    <Page Name="Options">
        <Text X="11" Y="80" Width="-11" Height="30" FontId="2" DisablePrefix="yes">#(loc.OptionsHeader)</Text>

        <!-- Component Selection Section -->
        <Text X="11" Y="115" Width="-11" Height="18" FontId="3">Select VIAME components to install:</Text>

        <!-- GPU/CUDA Support -->
        <Checkbox Name="CUDACheckbox" X="20" Y="138" Width="-30" Height="17" TabStop="yes" FontId="3">CUDA/cuDNN Support (Required for GPU acceleration)</Checkbox>
        <Text X="36" Y="155" Width="-40" Height="14" FontId="5">Enables GPU-accelerated processing with NVIDIA graphics cards</Text>

        <!-- PyTorch -->
        <Checkbox Name="PyTorchCheckbox" X="20" Y="175" Width="-30" Height="17" TabStop="yes" FontId="3">PyTorch + Deep Learning Libraries</Checkbox>
        <Text X="36" Y="192" Width="-40" Height="14" FontId="5">Includes MMDetection, Ultralytics, SAM2, and other ML frameworks</Text>

        <!-- Extra C++ Detectors -->
        <Checkbox Name="ExtraCPPCheckbox" X="20" Y="212" Width="-30" Height="17" TabStop="yes" FontId="3">Extra C++ Detectors (Darknet, SVM, PostgreSQL)</Checkbox>
        <Text X="36" Y="229" Width="-40" Height="14" FontId="5">Additional detection algorithms and database support</Text>

        <!-- GUI Section -->
        <Text X="11" Y="255" Width="-11" Height="18" FontId="3">Select GUI interfaces to install:</Text>

        <Checkbox Name="DIVECheckbox" X="20" Y="278" Width="-30" Height="17" TabStop="yes" FontId="3">DIVE GUI (Web-based annotation interface)</Checkbox>
        <Text X="36" Y="295" Width="-40" Height="14" FontId="5">Modern web interface for video annotation and review</Text>

        <Checkbox Name="VIVIACheckbox" X="20" Y="315" Width="-30" Height="17" TabStop="yes" FontId="3">VIVIA Interface (Desktop application with Qt/VTK)</Checkbox>
        <Text X="36" Y="332" Width="-40" Height="14" FontId="5">Traditional desktop GUI for visualization and analysis</Text>

        <!-- Advanced Section -->
        <Text X="11" Y="358" Width="-11" Height="18" FontId="3">Advanced options:</Text>

        <Checkbox Name="SEALCheckbox" X="20" Y="381" Width="-30" Height="17" TabStop="yes" FontId="3">SEAL Toolkit</Checkbox>
        <Text X="36" Y="398" Width="-40" Height="14" FontId="5">Specialized tools for seal detection and tracking</Text>

        <Checkbox Name="ModelsCheckbox" X="20" Y="418" Width="-30" Height="17" TabStop="yes" FontId="3">Pre-trained Models</Checkbox>
        <Text X="36" Y="435" Width="-40" Height="14" FontId="5">Download pre-trained detection and classification models</Text>

        <Checkbox Name="DevHeadersCheckbox" X="20" Y="455" Width="-30" Height="17" TabStop="yes" FontId="3">Development Headers</Checkbox>
        <Text X="36" Y="472" Width="-40" Height="14" FontId="5">Include and share folders for building against VIAME</Text>

        <!-- Model Addon Selection Section (auto-generated from CSV) -->
        <Text X="11" Y="498" Width="-11" Height="18" FontId="3">Additional model packages ({len(self.config.model_addons)} available):</Text>

{model_checkboxes_xml}

        <Button Name="OptionsOkButton" X="-91" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0">#(loc.OptionsOkButton)</Button>
        <Button Name="OptionsCancelButton" X="-11" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0">#(loc.OptionsCancelButton)</Button>
    </Page>

    <Page Name="Progress">
        <Text X="11" Y="80" Width="-11" Height="30" FontId="2">#(loc.ProgressHeader)</Text>
        <Text X="11" Y="121" Width="70" Height="17" FontId="3">#(loc.ProgressLabel)</Text>
        <Text Name="OverallProgressPackageText" X="85" Y="121" Width="-11" Height="17" FontId="3">#(loc.OverallProgressPackageText)</Text>
        <Progressbar Name="OverallCalculatedProgressbar" X="11" Y="143" Width="-11" Height="15" />
        <Button Name="ProgressCancelButton" X="-11" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0">#(loc.ProgressCancelButton)</Button>
    </Page>

    <Page Name="Modify">
        <Text X="11" Y="80" Width="-11" Height="30" FontId="2">#(loc.ModifyHeader)</Text>
        <Button Name="RepairButton" X="-171" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0" HideWhenDisabled="yes">#(loc.ModifyRepairButton)</Button>
        <Button Name="UninstallButton" X="-91" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0">#(loc.ModifyUninstallButton)</Button>
        <Button Name="ModifyCancelButton" X="-11" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0">#(loc.ModifyCloseButton)</Button>
    </Page>

    <Page Name="Success">
        <Text X="11" Y="80" Width="-11" Height="30" FontId="2">#(loc.SuccessHeader)</Text>
        <Button Name="LaunchButton" X="-91" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0" HideWhenDisabled="yes">#(loc.SuccessLaunchButton)</Button>
        <Text Name="SuccessRestartText" X="-11" Y="-51" Width="400" Height="34" FontId="3" HideWhenDisabled="yes">#(loc.SuccessRestartText)</Text>
        <Button Name="SuccessRestartButton" X="-91" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0" HideWhenDisabled="yes">#(loc.SuccessRestartButton)</Button>
        <Button Name="SuccessCancelButton" X="-11" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0">#(loc.SuccessCloseButton)</Button>
    </Page>

    <Page Name="Failure">
        <Text X="11" Y="80" Width="-11" Height="30" FontId="2">#(loc.FailureHeader)</Text>
        <Hypertext Name="FailureLogFileLink" X="11" Y="121" Width="-11" Height="42" FontId="3" TabStop="yes" HideWhenDisabled="yes">#(loc.FailureHyperlinkLogText)</Hypertext>
        <Hypertext Name="FailureMessageText" X="22" Y="163" Width="-11" Height="51" FontId="3" TabStop="yes" HideWhenDisabled="yes" />
        <Text Name="FailureRestartText" X="-11" Y="-51" Width="400" Height="34" FontId="3" HideWhenDisabled="yes">#(loc.FailureRestartText)</Text>
        <Button Name="FailureRestartButton" X="-91" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0" HideWhenDisabled="yes">#(loc.FailureRestartButton)</Button>
        <Button Name="FailureCloseButton" X="-11" Y="-11" Width="75" Height="23" TabStop="yes" FontId="0">#(loc.FailureCloseButton)</Button>
    </Page>
</Theme>
'''

        ui_file.write_text(theme_xml)
        logger.info(f"Generated options UI with {len(self.config.model_addons)} model checkboxes")

        return ui_file

    def run(self) -> None:
        """Run the installer generation process."""
        logger.info("=" * 60)
        logger.info("VIAME Installer Generator")
        logger.info("=" * 60)
        logger.info(f"Found {len(self.config.model_addons)} model addons in CSV")

        # Generate UI fragment first (doesn't require Girder)
        logger.info("\n--- Generating Model UI Fragment ---")
        self.generate_ui_fragment()

        # Initialize Girder connection
        self.initialize()

        # Download external installers
        logger.info("\n--- Downloading External Installers ---")
        for name, info in self.config.external_installers.items():
            try:
                self.download_external_installer(name, info)
            except Exception as e:
                logger.warning(f"Failed to download {name}: {e}")

        # Generate model data installers from CSV
        logger.info("\n--- Generating Model Data Installers ---")
        manager = multiprocessing.Manager()
        results = manager.dict()

        # Process models from CSV
        for model in self.config.model_addons:
            try:
                self.generate_model_installer(model, results)
            except Exception as e:
                logger.error(f"Failed to generate {model.name}: {e}")
                if not self.options.get('continue_on_error', True):
                    raise

        # Generate chain file
        logger.info("\n--- Generating WiX Chain File ---")
        if results:
            self.generate_chain_file(dict(results))
        else:
            logger.warning("No installers generated, skipping chain file")

        logger.info("\n" + "=" * 60)
        logger.info("Installer generation complete!")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = ArgumentParser(
        description="Generate VIAME network installer components"
    )
    parser.add_argument(
        "--remake-all", "--remake_all",
        action="store_true",
        default=False,
        help="Regenerate all installers even if they exist"
    )
    parser.add_argument(
        "-j", "--num-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: from config)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE,
        help=f"Configuration file path (default: {CONFIG_FILE})"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing if one installer fails"
    )
    parser.add_argument(
        "--ui-only",
        action="store_true",
        help="Only generate the UI theme file (no Girder auth required)"
    )
    parser.add_argument(
        "--addons-csv",
        type=Path,
        default=ADDONS_CSV_FILE,
        help=f"Model addons CSV file path (default: {ADDONS_CSV_FILE})"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config = InstallerConfig.load(args.config, args.addons_csv)

        if args.num_processes:
            config.num_processes = args.num_processes

        options = {
            'remake_all': args.remake_all,
            'continue_on_error': args.continue_on_error,
            'ui_only': args.ui_only
        }

        generator = InstallerGenerator(config, options)

        if args.ui_only:
            # Just generate UI, no Girder needed
            generator.generate_ui_fragment()
            logger.info("UI generation complete!")
        else:
            generator.run()

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except EnvironmentError as e:
        logger.error(f"Environment error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
