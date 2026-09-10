"""
Python-side plugin discover logic.
This is expected to be invoked by the C++ factory registration function.

"""

import importlib
import logging
import os
import sys
from typing import Iterable
from typing import List
from typing import Set
from typing import Type
from typing import TypeVar
from types import ModuleType

from kwiver.vital.plugins import Pluggable


# Before 3.8, we depend on importlib_metadata >=3.7.0, which is in parity with
# the python version 3.10+ `importlib.metadata.entry_points`.
# This comparison IS NOT CHAINED on purpose to support mypy compatibility.
# noinspection PyChainedComparisons
if sys.version_info >= (3, 10):
    import importlib.metadata as metadata

    # entry_points() returns an EntryPoints object from 3.10 on, and its
    # dict-style .get() was removed outright in 3.12, so select by group.
    def get_ns_entrypoints(ns: str) -> Iterable["metadata.EntryPoint"]:
        return metadata.entry_points(group=ns)

elif sys.version_info >= (3, 8):
    import importlib.metadata as metadata

    def get_ns_entrypoints(ns: str) -> Iterable["metadata.EntryPoint"]:
        return metadata.entry_points().get(ns, ())

else:
    import importlib_metadata as metadata

    def get_ns_entrypoints(ns: str) -> Iterable["metadata.EntryPoint"]:
        return metadata.entry_points(group=ns)


# TODO: Change to use kwiver logging when bindings added.
#       Theoretically straight forward...
LOG = logging.getLogger(__name__)

# Environment variable *PATH separator for the current platform.
OS_ENV_PATH_SEP = os.pathsep

# String name of the namespace from which we query for entry-points.
PLUGIN_NAMESPACE = "kwiver.python_plugins"
PLUGIN_ENV_VAR = "KWIVER_PYTHON_PLUGIN_PATH"

# Type variable for some subtype of the Pluggable abstract class.
P = TypeVar("P", bound=Pluggable)
# Generic type variable for generics annotations below.
T = TypeVar("T")


class NotAModuleError(Exception):
    """
    Exception for when the `discover_via_entrypoint_extensions` function
    found an entrypoint that was *not* a module specification.
    """


def _collect_types_in_module(module: ModuleType) -> Set[Type]:
    """
    Common method of returning a set of class types defined in a python module.

    If you happened to want to dynamically reload the types in a module that is
    updated during runtime:

    .. code-block:: python

       module = importlib.reload(module)
       _collect_types_in_module(module)
    """
    type_set: Set[Type] = set()
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type):
            type_set.add(attr)
    return type_set


def import_via_entrypoint_extensions(entrypoint_ns: str) -> Set[ModuleType]:
    """
    Discover and import modules exposed through the entry-point extensions
    defined for the given namespace by installed python packages, returning the
    module instances.

    Other installed python packages may define one or more extensions for
    a namespace, as specified by `ns`, in their "setup.py" (or equivalent
    section).
    This should be a single or list of extensions that specify modules within
    the installed package where plugins for export are implemented.

    Currently, this method only accepts extensions that export a module as
    opposed to specifications of a specific attribute in a module.
    This is due to later type discovery making use of `__subclasses__`
    not necessarily honoring the
    selectivity that specific attribute specification provides
    (Looking at you `__subclasses__`...).

    For example, as a single specification string::

        ...
        entry_points={
            "kwiver.python_plugins": "my_package = my_package.plugins"
        ]
        ...

    Or in list form of multiple specification strings::

        ...
        entry_points = {
            "kwiver.python_plugins": [
                "my_package_mode_1 = my_package.mode_1.plugins",
                "my_package_mode_2 = my_package.mode_2.plugins",
            ]
        }
        ...

    :param entrypoint_ns: The name of the entry-point mapping in  to look for
        extensions under.

    :raises NotAModuleError: When an entry-point extension specified something
        more specific than the module level.
    :raises ModuleNotFoundError: When one or more module paths specified in the
        given entry-point extension specifications that are not importable.

    :return: Set of discovered modules from the extension specifications under
        the specified entry-point namespace.
    """
    mod_set: Set[ModuleType] = set()
    for entry_point in get_ns_entrypoints(entrypoint_ns):
        m = entry_point.load()
        if not isinstance(m, ModuleType):
            raise NotAModuleError(
                f"An entrypoint with key '{entry_point.name}' and value "
                f"'{entry_point.value}' did not specify a module (got an "
                f"object of type `{type(m).__name__}` instead): {m}"
            )
        else:
            mod_set.add(m)
    return mod_set


def import_via_env_var(env_var: str) -> Set[ModuleType]:
    """
    Discover, import and return python-importable modules specified in the
    given environment variable.

    We expect the given environment variable to define zero or more python
    module paths from which to yield all contained type definitions (i.e.
    things that descent from `type`). If there is an empty path element, it is
    skipped (e.g. "foo::bar:baz" will only attempt importing `foo`, `bar` and
    `baz` modules).

    These python module paths should be separated with the same separator as
    would be used in the PYTHONPATH environment variable specification.

    If a module defines no class types, then no types are included from that
    source for return.

    An expected use-case for this discovery method is for modules
    that are not installed but otherwise accessible via the python search path.
    E.g. local modules, modules accessible through PYTHONPATH search path
    modification, modules accessible through `sys.path` modification.

    Any errors raised from attempting to import a module are propagated upward.

    :param env_var: The name of the environment variable to read from.

    :raises ModuleNotFoundError: When one or more module paths specified in the
        given environment variable are not importable.

    :return: Set of discovered types from the modules specified in the
        environment variable's contents.
    """
    mod_set: Set[ModuleType] = set()
    env_var_paths = os.environ.get(env_var, "").split(OS_ENV_PATH_SEP)
    llevel = 1
    # If no value, and empty string splits into `[""]`.
    if env_var_paths == [""]:
        LOG.log(
            llevel,
            f"Environment variable `{env_var}` not defined or did not "
            f"contain any module paths.",
        )
    for path in env_var_paths:
        # Skip empty strings
        if path:
            # May raise ModuleNotFoundError if `path` is not a valid,
            # importable module path.
            m = importlib.import_module(path)
            LOG.log(
                llevel,
                f"For environment variable `{env_var}`, imported module "
                f"path `{path}`.",
            )
            mod_set.add(m)
    return mod_set


def traverse_subclasses(type_: Type[T]) -> Set[Type[T]]:
    """
    Traverse and return *all* subclass types currently in the import-scope of
    the given type the tree of subclasses underneath Pluggable.

    :return: Set of concrete Pluggable sub-types.
    """
    # __subclasses__ only returns *immediate* subclasses, i.e. one level.
    # To get nested subclasses we'll have to do some graph traversal.
    class_set = set()

    # Use a list (stack behavior) to track the descendant classes of
    # `interface_type`. Depth- vs. Breadth-first search should not matter here,
    # so just using just using lists here for theoretically more optimal array
    # caching.
    candidates: List[Type[T]] = type_.__subclasses__()
    while candidates:
        class_type = candidates.pop()
        class_set.add(class_type)
        candidates.extend(class_type.__subclasses__())
    return class_set


def is_concrete_pluggable(t: Type[P]) -> bool:
    """
    Test if the given Pluggable-inheriting type is a concrete, instantiable
    type.

    Currently, this is achieved by property checking the type for
    "concrete-looking" attributes. This is not robust...

    :param t: Type to check,
    :return: True if the given type is considered "concrete" and False
        otherwise.
    """
    # TODO: Would love to rely on better introspection than duck-type checks as
    #       this would need to be co-updated with binding definition...
    if (
        hasattr(t, "interface_name")
        and hasattr(t, "from_config")
        and hasattr(t, "get_default_config")
    ):
        return True
    return False


def _get_concrete_pluggable_types() -> List[Type[Pluggable]]:
    """
    Get all known python-implemented concrete implementations of the Pluggable
    interface.

    The results of this will be used for plugin registration
    """
    import_via_entrypoint_extensions(PLUGIN_NAMESPACE)
    import_via_env_var(PLUGIN_ENV_VAR)
    p_type_set = traverse_subclasses(Pluggable)
    concrete_types = [p_t for p_t in p_type_set if (is_concrete_pluggable(p_t))]

    # Also include legacy algorithms registered through algorithm_factory
    try:
        from kwiver.vital.algo.algorithm_factory import get_registered_algorithms

        legacy_algos = get_registered_algorithms()
        for algo_cls in legacy_algos:
            if algo_cls not in concrete_types:
                concrete_types.append(algo_cls)
    except Exception:
        pass  # If legacy import fails, continue with discovered types

    return concrete_types
