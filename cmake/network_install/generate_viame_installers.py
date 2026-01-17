#!/usr/bin/env python3
"""
VIAME Network Installer Generator

Generates MSI installers for VIAME components and model data packages,
uploads them to Girder, and creates WiX chain files for the network installer.

Usage:
    python generate_viame_installers.py [options]

Environment Variables:
    GIRDER_USER     - Girder username for authentication
    GIRDER_PASSWORD - Girder password for authentication
    GIRDER_API_KEY  - Alternative: Girder API key (instead of user/password)
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import shutil
import signal
import sys
import urllib.request
from argparse import ArgumentParser
from dataclasses import dataclass
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
CONFIG_FILE = SCRIPT_DIR / "installer_config.json"


@dataclass
class InstallerConfig:
    """Configuration for the installer generator."""
    viame_version: dict[str, int]
    data_version: dict[str, int]
    girder_api_url: str
    girder_data_url: str
    components: dict[str, dict]
    model_data: dict[str, dict]
    external_installers: dict[str, dict]
    num_processes: int
    girder_folder_id: str | None = None

    @classmethod
    def load(cls, config_path: Path = CONFIG_FILE) -> 'InstallerConfig':
        """Load configuration from JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            data = json.load(f)

        return cls(
            viame_version=data['viame_version'],
            data_version=data['data_version'],
            girder_api_url=data['girder']['api_url'],
            girder_data_url=data['girder']['data_url'],
            components=data.get('components', {}),
            model_data=data.get('model_data', {}),
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


# CMake template for data package installers
CMAKE_TEMPLATE = """
cmake_minimum_required(VERSION 3.16)
project({data_name}Data NONE)

if(NOT EXISTS ${{CMAKE_BINARY_DIR}}/{data_name}.zip)
  file(DOWNLOAD {download_url}
    ${{CMAKE_BINARY_DIR}}/{data_name}.zip
    SHOW_PROGRESS
    STATUS download_status)
  list(GET download_status 0 status_code)
  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "Download failed: ${{download_status}}")
  endif()
endif()

install(FILES ${{CMAKE_BINARY_DIR}}/{data_name}.zip
  DESTINATION data)

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

    def generate_model_installer(self, name: str, model_info: dict,
                                  results: dict) -> None:
        """Generate an MSI installer for a model data package."""
        version_major = self.config.data_version['major']
        version_minor = self.config.data_version['minor']
        target_name = f"{name}Data-{version_major}.{version_minor}-win32.msi"

        # Check if already exists
        if self.upload_folder_id and not self.options.get('remake_all', False):
            existing = self.girder.find_item(self.upload_folder_id, target_name)
            if existing:
                logger.info(f"Skipping {name} - already exists")
                existing['itemId'] = existing['_id']
                results[target_name] = existing
                return

        logger.info(f"Generating installer for {name}...")

        download_url = (
            f"{self.config.girder_data_url}/item/{model_info['girder_id']}/download"
        )

        try:
            with TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Write CMakeLists.txt
                cmake_content = CMAKE_TEMPLATE.format(
                    data_name=name,
                    download_url=download_url,
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
                    raise RuntimeError(f"No MSI file generated for {name}")

                installer_path = msi_files[0]
                logger.info(f"Generated: {installer_path.name} "
                           f"({installer_path.stat().st_size / 1024 / 1024:.1f} MB)")

                # Upload to Girder
                if self.upload_folder_id:
                    result = self.girder.upload_file(
                        self.upload_folder_id, installer_path
                    )
                    results[installer_path.name] = result
                    logger.info(f"Uploaded {installer_path.name} to Girder")

                # Copy to script directory
                dest_path = SCRIPT_DIR / installer_path.name
                shutil.copy2(installer_path, dest_path)
                logger.info(f"Copied to {dest_path}")

        except CalledProcessError as e:
            logger.error(f"Build failed for {name}: {e.stderr.decode() if e.stderr else e}")
            raise
        except Exception as e:
            logger.error(f"Error generating installer for {name}: {e}")
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

    def generate_chain_file(self, installer_results: dict) -> Path:
        """Generate the WiX chain file for all installers."""
        chain_file = SCRIPT_DIR / "VIAME_Chain_File.wxs"

        logger.info(f"Generating chain file: {chain_file}")

        content = ["<Include>"]

        for installer_name, installer_data in installer_results.items():
            # Extract base name for condition
            base_name = installer_name.split("-")[0]
            package_id = f"{base_name}Data"
            condition = f"{base_name}ModelsCheckbox=1"

            download_url = (
                f"{self.config.girder_api_url}/item/{installer_data['itemId']}/download"
            )

            content.append(WIX_PACKAGE_TEMPLATE.format(
                package_id=package_id,
                file_name=installer_name,
                download_url=download_url,
                display_name=installer_name,
                condition=condition
            ))

        content.append("</Include>")

        chain_file.write_text("\n".join(content))
        logger.info(f"Chain file generated with {len(installer_results)} packages")

        return chain_file

    def run(self) -> None:
        """Run the installer generation process."""
        logger.info("=" * 60)
        logger.info("VIAME Installer Generator")
        logger.info("=" * 60)

        # Initialize Girder connection
        self.initialize()

        # Download external installers
        logger.info("\n--- Downloading External Installers ---")
        for name, info in self.config.external_installers.items():
            try:
                self.download_external_installer(name, info)
            except Exception as e:
                logger.warning(f"Failed to download {name}: {e}")

        # Generate model data installers
        logger.info("\n--- Generating Model Data Installers ---")
        manager = multiprocessing.Manager()
        results = manager.dict()

        # Process models (can be parallelized if needed)
        for name, info in self.config.model_data.items():
            try:
                self.generate_model_installer(name, info, results)
            except Exception as e:
                logger.error(f"Failed to generate {name}: {e}")
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

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        config = InstallerConfig.load(args.config)

        if args.num_processes:
            config.num_processes = args.num_processes

        options = {
            'remake_all': args.remake_all,
            'continue_on_error': args.continue_on_error
        }

        generator = InstallerGenerator(config, options)
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
