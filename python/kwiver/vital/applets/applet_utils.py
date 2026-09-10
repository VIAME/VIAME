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
Utilities for creating and registering Python KWIVER applets.
"""

from kwiver.vital.applets import KwiverApplet
from kwiver.vital.plugins import register


def register_applet(applet_class, plugin_name=None, description=""):
    """
    Register a Python applet with the KWIVER plugin system.

    This function simplifies the registration of Python-based KWIVER applets,
    making them discoverable and runnable through the kwiver command-line tool.

    Args:
        applet_class: A class that inherits from KwiverApplet
        plugin_name: Optional name for the plugin. If None, uses the class name
        description: Optional description of the applet

    Example:
        >>> from kwiver.vital.applets import KwiverApplet
        >>> from kwiver.vital.applets.applet_utils import register_applet
        >>>
        >>> class MyApplet(KwiverApplet):
        >>>     def run(self):
        >>>         print("Hello from my applet!")
        >>>         return 0
        >>>
        >>> def register_plugins():
        >>>     register_applet(MyApplet, "my-applet", "My custom applet")
    """
    if not issubclass(applet_class, KwiverApplet):
        raise TypeError(f"{applet_class.__name__} must inherit from KwiverApplet")

    if plugin_name is None:
        plugin_name = applet_class.__name__

    # Use the KWIVER plugin registration system
    register.plugin_factory(
        plugin_type="kwiver_applet",
        plugin_name=plugin_name,
        factory_func=applet_class,
        description=description,
    )


__all__ = ["register_applet"]
