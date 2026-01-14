# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Python Process Registration Module for VIAME Examples.

This module registers Python-based sprokit processes with the KWIVER pipeline
system. When the pipeline loads this module, it makes the example processes
available for use in pipeline configurations.

To add your own process:
    1. Import your process class at the top of this file
    2. Add a process_factory.add_process() call in __sprokit_register__()

The registration function is automatically called by the KWIVER module loader.
"""

from viame.examples import hello_world_detector
from viame.examples import hello_world_filter


def __sprokit_register__():
    """
    Register Python processes with the KWIVER pipeline system.

    This function is called automatically by the KWIVER module loader when
    the pipeline system initializes. It registers all Python processes
    defined in this package so they can be used in pipeline configurations.

    The module loading check prevents duplicate registration if this module
    is imported multiple times.
    """
    from kwiver.sprokit.pipeline import process_factory

    # Unique module identifier - prevents double-loading
    module_name = 'python:examples.example_processes'

    # Skip if already loaded (prevents duplicate registration)
    if process_factory.is_process_module_loaded( module_name ):
        return

    # Register each process with:
    #   - process_name: Name used in pipeline files (e.g., ":: hello_world_detector")
    #   - description: Brief description shown in process listings
    #   - class: The Python class that implements the process
    process_factory.add_process(
        'hello_world_detector',
        'Example detector that logs a message for each image',
        hello_world_detector.hello_world_detector )

    process_factory.add_process(
        'hello_world_filter',
        'Example filter that logs a message and passes images through',
        hello_world_filter.hello_world_filter )

    # Mark module as loaded to prevent duplicate registration
    process_factory.mark_process_module_as_loaded( module_name )
