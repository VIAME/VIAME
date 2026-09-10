#!/usr/bin/env python
# ckwg +29
# Copyright 2025 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF

"""
Example Python KWIVER Applet

This example demonstrates how to create a custom KWIVER applet in Python
using the Python bindings for the applet framework.
"""

import sys
from kwiver.vital.applets import KwiverApplet, register_applet
from kwiver.vital.config import empty_config


class ExamplePythonApplet(KwiverApplet):
    """
    A simple example applet that demonstrates the basic structure
    of a Python-based KWIVER applet.
    """

    def __init__(self):
        super().__init__()
        self.input_file = None
        self.output_file = None
        self.verbose = False

    def add_command_options(self):
        """
        Add command-line options for this applet.

        Note: The current implementation uses argparse-style parsing
        in Python rather than the C++ cxxopts library.
        """
        # This would be implemented when argparse binding is added
        # For now, we'll process args manually in run()
        pass

    def set_configuration(self, config):
        """
        Set configuration for this applet.

        Args:
            config: ConfigBlock containing applet configuration
        """
        if config.has_value("input"):
            self.input_file = config.get_value("input")

        if config.has_value("output"):
            self.output_file = config.get_value("output")

        if config.has_value("verbose"):
            self.verbose = config.get_value_bool("verbose")

    def get_configuration(self):
        """
        Get the current configuration for this applet.

        Returns:
            ConfigBlock containing current applet configuration
        """
        config = empty_config()

        if self.input_file:
            config.set_value("input", self.input_file)

        if self.output_file:
            config.set_value("output", self.output_file)

        config.set_value("verbose", str(self.verbose))

        return config

    def run(self):
        """
        Main entry point for the applet.

        Returns:
            int: Application return code (0 for success, non-zero for error)
        """
        print("=" * 60)
        print("Example Python KWIVER Applet")
        print("=" * 60)

        # Get command-line arguments
        args = self.applet_args()

        print(f"\nApplet name: {self.applet_name()}")
        print(f"Arguments: {args}")

        # Parse simple arguments for demonstration
        if "--help" in args or "-h" in args:
            self.print_help()
            return 0

        if "--verbose" in args or "-v" in args:
            self.verbose = True

        # Process input/output files if provided
        for i, arg in enumerate(args):
            if arg == "--input" or arg == "-i":
                if i + 1 < len(args):
                    self.input_file = args[i + 1]
            elif arg == "--output" or arg == "-o":
                if i + 1 < len(args):
                    self.output_file = args[i + 1]

        if self.verbose:
            print("\nConfiguration:")
            config = self.get_configuration()
            for key in config.available_values():
                print(f"  {key} = {config.get_value(key)}")

        if self.input_file:
            print(f"\nProcessing input file: {self.input_file}")
            # Add your processing logic here

        if self.output_file:
            print(f"Writing output to: {self.output_file}")
            # Add your output logic here

        print("\nApplet completed successfully!")
        return 0

    def print_help(self):
        """Print help message for this applet."""
        help_text = """
Usage: kwiver example-python-applet [options]

A simple example Python applet for KWIVER.

Options:
  -h, --help              Show this help message
  -v, --verbose           Enable verbose output
  -i, --input FILE        Input file to process
  -o, --output FILE       Output file to write

Example:
  kwiver example-python-applet -v -i input.txt -o output.txt
"""
        print(self.wrap_text(help_text))


def register_plugins():
    """
    Registration function called by the KWIVER plugin system.

    This function should be specified as an entry point when
    packaging your Python applet as a plugin.
    """
    register_applet(
        ExamplePythonApplet,
        plugin_name="example-python-applet",
        description="Example Python-based KWIVER applet for demonstration",
    )


if __name__ == "__main__":
    # Allow running the applet directly for testing
    applet = ExamplePythonApplet()
    # Simulate initialization that would normally be done by kwiver runner
    from kwiver.vital.applets import AppletContext

    context = AppletContext()
    context.applet_name = "example-python-applet"
    context.argv = sys.argv[1:]
    # Note: Full initialization would require more setup
    sys.exit(applet.run())
