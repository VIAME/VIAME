# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Pytest configuration for VIAME tests.

This sets up the import path aliases so that tests can import from
viame.pytorch, viame.core, etc. when running against the source tree.
"""

import sys
from pathlib import Path


def pytest_configure(config):
    """Set up import path aliases for viame packages."""
    # Get the source root directory
    src_root = Path(__file__).parent.parent

    # Create the viame namespace package structure in sys.modules
    import types

    # Create viame package if not already present
    if 'viame' not in sys.modules:
        viame = types.ModuleType('viame')
        viame.__path__ = []
        sys.modules['viame'] = viame
    else:
        viame = sys.modules['viame']

    # Map viame.pytorch to plugins/pytorch
    pytorch_path = src_root / 'plugins' / 'pytorch'
    if pytorch_path.exists():
        if 'viame.pytorch' not in sys.modules:
            # Add to viame's path
            viame.__path__.append(str(pytorch_path.parent))

            # Import the module and add it
            if str(pytorch_path) not in sys.path:
                sys.path.insert(0, str(pytorch_path.parent))

            # Create viame.pytorch module alias
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'viame.pytorch',
                pytorch_path / '__init__.py',
                submodule_search_locations=[str(pytorch_path)]
            )
            if spec and spec.loader:
                pytorch_module = importlib.util.module_from_spec(spec)
                pytorch_module.__path__ = [str(pytorch_path)]
                sys.modules['viame.pytorch'] = pytorch_module

                # Now we can import submodules - add the path for them
                sys.path.insert(0, str(pytorch_path))

    # Map viame.core to plugins/core
    core_path = src_root / 'plugins' / 'core'
    if core_path.exists():
        if 'viame.core' not in sys.modules:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'viame.core',
                core_path / '__init__.py',
                submodule_search_locations=[str(core_path)]
            )
            if spec and spec.loader:
                core_module = importlib.util.module_from_spec(spec)
                core_module.__path__ = [str(core_path)]
                sys.modules['viame.core'] = core_module
                sys.path.insert(0, str(core_path))
