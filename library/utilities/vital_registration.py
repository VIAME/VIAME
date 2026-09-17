# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

"""Registration helper for python algorithm implementations.

KWIVER's pluggable system discovers python implementations by walking the
subclasses of ``Pluggable`` (see ``viame.plugins.discovery``).  A class
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
        algorithm_class.get_default_config = classmethod(_default_config)

    return algorithm_class


def _default_config(cls, cb):
    """Fill ``cb`` with what a default instance of ``cls`` is configured with.

    This returned ``None`` and set nothing until it was noticed what that
    cost. A python implementation declares its options in
    ``get_configuration()``, which is where pipelines read them from, so the
    options work -- but nothing that *asks* what an implementation provides
    could see them. ``registry-dump`` listed no config keys at all for every
    python algorithm, and ``compare_registry.py`` skips an entry it could not
    introspect, so the compatibility baseline stopped checking the keys and
    defaults of each algorithm at the moment it was ported from C++ to
    python. 143 keys across 24 implementations were in that position.

    A default instance's ``get_configuration()`` *is* the default config, so
    there is nothing to duplicate: build one and merge it.
    """
    try:
        instance = cls()
    except Exception as error:  # noqa: BLE001 - reported, not raised
        raise RuntimeError(
            "{} cannot be constructed with no arguments, so its default "
            "configuration cannot be read: {}".format(cls.__name__, error)
        ) from error

    cb.merge_config(instance.get_configuration())
