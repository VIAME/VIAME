# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""Registration helper for python algorithm implementations.

KWIVER's pluggable system discovers python implementations by walking the
subclasses of ``Pluggable`` (see ``kwiver.vital.plugins.discovery``).  A class
is only picked up if it looks concrete -- that is, if it exposes
``interface_name``, ``from_config`` and ``get_default_config`` -- and it
registers under ``plugin_name()`` if that is defined, otherwise under its
``__name__``.

Most VIAME implementations only define ``__init__`` and the interface method,
and the name a pipeline refers to them by (``netharn``, ``ocv_windowed``, ...)
is not their class name.  This helper supplies the missing pieces so the
implementation is discovered under the intended name, without reaching for any
of the pre-2.0 registration API.
"""

__all__ = ["register_vital_algorithm"]


def register_vital_algorithm(algorithm_class, implementation_name, description):
    """Make ``algorithm_class`` discoverable as ``implementation_name``.

    Args:
        algorithm_class: Implementation class, deriving from a vital algorithm
            interface such as ``ImageObjectDetector``.
        implementation_name: Name pipelines use to select this implementation.
        description: Human-readable description.
    """
    # python_plugin_factory prefers plugin_name() over __name__.
    if "plugin_name" not in vars(algorithm_class):
        algorithm_class.plugin_name = staticmethod(lambda _n=implementation_name: _n)
    if "plugin_description" not in vars(algorithm_class):
        algorithm_class.plugin_description = staticmethod(lambda _d=description: _d)

    # is_concrete_pluggable() duck-types on these two; supply the obvious
    # defaults for implementations that are constructed without arguments and
    # take their settings through set_configuration().
    if "from_config" not in vars(algorithm_class):
        algorithm_class.from_config = classmethod(lambda cls, c: cls())
    if "get_default_config" not in vars(algorithm_class):
        algorithm_class.get_default_config = classmethod(lambda cls, c: None)

    return algorithm_class
