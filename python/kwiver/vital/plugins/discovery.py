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

    # Declared implementations, which have not been imported. A package that
    # declares gets its entries from here; one that does not is imported by
    # `module_loader` as before and shows up in the subclass walk above.
    declared_names = set()
    for proxy in declared_pluggable_types():
        key = (proxy.interface_name(), proxy.plugin_name())
        if key in declared_names:
            continue
        declared_names.add(key)
        concrete_types.append(proxy)

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


# ----------------------------------------------------------------------------
# Lazy declarations
#
# A package may say what it provides without importing the modules that
# provide it:
#
#     __vital_algorithm_declarations__ = [
#         ( "track_objects", "srnn", "Structural RNN tracker",
#           "viame.object_trackers.pytorch.srnn_tracker:SRNNTracker" ),
#     ]
#
# The alternative, and what every package did before P8-T10, is to import
# each implementation module so that its class exists for `traverse_subclasses`
# to find. That is the only reason the imports happened -- and importing
# `viame.pytorch` imports torch, 1.4 seconds, on every `viame` command that
# touched the plugin system, including `viame runner --help`.
#
# A declaration is turned into a proxy type that looks concrete to
# `python_plugin_factory` and defers: the real module is imported the first
# time something asks for an instance.

# The packages that ship with VIAME. Before P8-T10 this list lived in
# `setup_viame.sh` as sixteen `export SPROKIT_PYTHON_MODULES=` lines, which
# meant the runtime could not find its own plugins unless a shell script had
# run first -- and that a name in it could be wrong without anything saying
# so. Four of the sixteen were: `kwiver.arrows.core`, `kwiver.arrows.python`,
# `kwiver.sprokit.processes.pytorch` and `kwiver.sprokit.tests.processes` are
# not installed by this tree under any option, and are not here.
#
# A package here that this build did not produce is simply not importable and
# is skipped with a debug line, so the list does not have to know which
# options were on: `viame.tensorflow` and `viame.colmap` come and go with
# theirs.
BUILTIN_PLUGIN_PACKAGES = (
    "kwiver.sprokit.processes",
    "kwiver.sprokit.schedulers",
    "viame.classifiers",
    "viame.classifiers.onnx",
    "viame.classifiers.netharn",
    "viame.classifiers.rf_detr",
    "viame.classifiers.sleap",
    "viame.descriptors.torchvision",
    "viame.file_io",
    "viame.image_processing",
    "viame.measurement",
    "viame.measurement.stereo",
    "viame.measurement.torchvision",
    "viame.object_detectors",
    "viame.object_detectors.onnx",
    "viame.object_detectors.detectron2",
    "viame.object_detectors.huggingface",
    "viame.object_detectors.learn",
    "viame.object_detectors.litdet",
    "viame.object_detectors.mit_yolo",
    "viame.object_detectors.mmdet",
    "viame.object_detectors.netharn",
    "viame.object_detectors.rf_detr",
    "viame.object_detectors.ultralytics",
    "viame.object_trackers",
    "viame.object_trackers.mdnet",
    "viame.object_trackers.pytorch",
    "viame.object_trackers.sam3",
    "viame.object_trackers.siammask",
    "viame.video_io",
    "viame.colmap",
    "viame.examples",
    "viame.onnx",
    "viame.opencv",
    "viame.pytorch",
    "viame.segmentation",
    "viame.segmentation.sam2",
    "viame.segmentation.sam3",
    "viame.training",
    "viame.tensorflow",
)

# Add-ons name their packages here. It replaces `SPROKIT_PYTHON_MODULES`, and
# unlike it is unset in a normal install: it is for packages VIAME does not
# ship, which is the only thing an environment variable was ever needed for.
PLUGIN_PACKAGES_ENV_VAR = "VIAME_PYTHON_PLUGINS"

# What the above resolved to, set by `load_python_modules` once it has run so
# that `registry-dump` can report the packages that actually contributed
# rather than the ones somebody asked for.
LOADED_PACKAGES_ENV_VAR = "VIAME_PYTHON_PLUGINS_LOADED"

# Set by `registry-dump --introspect`: answer questions that need the
# implementation imported, rather than refusing them.
INTROSPECT_ENV_VAR = "VIAME_INTROSPECT_PLUGINS"

DECLARATIONS_ATTR = "__vital_algorithm_declarations__"


class _NotIntrospectable(Exception):
    """Raised when answering would mean importing the implementation."""


def _resolve(import_path: str) -> Type:
    """`"package.module:Class"` to the class, importing the module."""
    module_name, _, class_name = import_path.partition(":")
    if not class_name:
        raise ValueError(
            "declaration import path needs 'module:Class', got "
            f"{import_path!r}"
        )
    module = importlib.import_module(module_name)

    # The class is not usable straight out of the module. A VIAME
    # implementation defines `__init__` and the interface method and nothing
    # else; `register_vital_algorithm`, called from the module's own
    # `__vital_algorithm_register__`, is what attaches `from_config`,
    # `get_default_config` and the plugin name. Scanning used to call it as a
    # side effect of finding the module, so a declaration has to call it
    # here -- otherwise the class arrives without the three methods the
    # factory needs, and the failure is an AttributeError at the moment
    # somebody tries to build one.
    registrar = getattr(module, "__vital_algorithm_register__", None)
    if registrar is not None:
        registrar()

    return getattr(module, class_name)


def _proxy_for(interface: str, name: str, description: str,
               import_path: str) -> Type:
    """A stand-in for a declared implementation that has not been imported.

    It answers the three questions registration asks -- which interface, what
    name, what description -- from the declaration, and imports only when
    asked to build something.
    """
    state = {"real": None}

    def real():
        if state["real"] is None:
            state["real"] = _resolve(import_path)
        return state["real"]

    def from_config(cls, cb):
        return real().from_config(cb)

    def get_default_config(cls, cb):
        # The one question a declaration cannot answer without importing:
        # the default config lives in the class. Everyday callers get the
        # refusal, because `registry-dump` asks every factory and importing
        # all of them costs about three seconds, most of it torch.
        #
        # The compatibility baseline needs the real answer, though -- a
        # config key that stops existing is exactly what it exists to catch
        # -- so `viame registry-dump --introspect` sets this and pays.
        if os.environ.get(INTROSPECT_ENV_VAR):
            return real().get_default_config(cb)

        raise _NotIntrospectable(
            f"'{name}' is declared lazily; its defaults would require "
            f"importing {import_path.split(':')[0]}. Pass --introspect to "
            f"registry-dump to import and read them."
        )

    return type(
        name,
        (),
        {
            "__doc__": description,
            "interface_name": staticmethod(lambda _i=interface: _i),
            "plugin_name": staticmethod(lambda _n=name: _n),
            "plugin_description": staticmethod(lambda _d=description: _d),
            "from_config": classmethod(from_config),
            "get_default_config": classmethod(get_default_config),
            "_viame_import_path": import_path,
        },
    )


def plugin_packages() -> List[str]:
    """Every package to ask for plugins: the built-ins, then any add-ons."""
    packages: List[str] = list(BUILTIN_PLUGIN_PACKAGES)

    for entry in os.environ.get(PLUGIN_PACKAGES_ENV_VAR, "").split(
            OS_ENV_PATH_SEP):
        if entry and entry not in packages:
            packages.append(entry)

    return packages


def _import_package(package: str):
    """The package, or None if it is not built into this install.

    Importing the package itself is cheap so long as its `__init__` only
    declares -- which is the whole discipline this depends on. A package that
    imports its implementations at the top of `__init__` pays for them here
    exactly as it did before.
    """
    try:
        return importlib.import_module(package)
    except ImportError as error:
        LOG.debug(f"Package {package!r} is not importable: {error}")
        return None


def package_declarations(package: str) -> List[tuple]:
    """A package's algorithm declarations, or an empty list if it has none."""
    module = _import_package(package)

    if module is None:
        return []

    return list(getattr(module, DECLARATIONS_ATTR, ()) or ())


def package_declares(package: str) -> bool:
    """Whether a package names what it provides rather than being scanned.

    Either kind of declaration counts: a package may ship only processes
    (`viame.examples`, `viame.colmap`) or only algorithms, and scanning one
    of those would import every module in it to find what it already said.

    So does an **empty** one. A package whose implementations have all moved
    elsewhere -- `viame.opencv` after P2-T05 -- declares `[]`, which says
    "nothing here", not "scan me". Testing the lists for truth read it as the
    second, and the scan then imported whatever modules were still on disk in
    that package: in an install prefix that only ever gains files, that was
    the pre-move `watershed_segmenter.py`, and `ocv_watershed` registered
    twice.
    """
    module = _import_package(package)

    if module is None:
        return False

    return (getattr(module, DECLARATIONS_ATTR, None) is not None
            or getattr(module, PROCESS_DECLARATIONS_ATTR, None) is not None)


def declared_pluggable_types() -> List[Type]:
    """Proxy types for everything the declaring packages declare."""
    proxies: List[Type] = []

    for package in plugin_packages():
        for declaration in package_declarations(package):
            try:
                interface, name, description, import_path = declaration
            except (TypeError, ValueError):
                LOG.warning(
                    f"{package}: ignoring malformed declaration "
                    f"{declaration!r}; expected "
                    "( interface, name, description, 'module:Class' )"
                )
                continue

            proxies.append(
                _proxy_for(interface, name, description, import_path))

    return proxies


# ----------------------------------------------------------------------------
# Process declarations
#
# The same idea as `__vital_algorithm_declarations__`, for sprokit processes:
#
#     __sprokit_process_declarations__ = [
#         ( "image_viewer", "Display input image and delay",
#           "viame.video_io.image_viewer:ImageViewer" ),
#     ]
#
# A process does not register through the subclass walk -- it calls
# `process_factory.add_process( name, description, ctor )` -- so the module
# had to be imported for the call to happen. The ctor is only ever *called*,
# though, so a function that imports and constructs is as good as the class
# and costs nothing until a pipeline actually wants one.

PROCESS_DECLARATIONS_ATTR = "__sprokit_process_declarations__"


def _lazy_process_ctor(import_path: str):
    """A process constructor that imports its module when first called."""

    def construct(config):
        return _resolve_plain(import_path)(config)

    return construct


def _resolve_plain(import_path: str):
    """`"package.module:Name"` to the object, importing the module.

    Unlike `_resolve`, no registrar is called: a process module registers by
    being imported and its class needs nothing attached to it.
    """
    module_name, _, attr = import_path.partition(":")
    if not attr:
        raise ValueError(
            f"declaration import path needs 'module:Name', got {import_path!r}"
        )
    return getattr(importlib.import_module(module_name), attr)


def register_declared_processes(package: str) -> int:
    """Register a package's declared processes lazily. Returns how many."""
    module = _import_package(package)

    if module is None:
        return 0

    declarations = getattr(module, PROCESS_DECLARATIONS_ATTR, ()) or ()
    if not declarations:
        return 0

    from kwiver.sprokit.pipeline import process_factory

    registered = 0
    for declaration in declarations:
        try:
            name, description, import_path = declaration
        except (TypeError, ValueError):
            LOG.warning(
                f"{package}: ignoring malformed process declaration "
                f"{declaration!r}; expected "
                "( name, description, 'module:Class' )"
            )
            continue

        process_factory.add_process(
            name, description, _lazy_process_ctor(import_path))
        registered += 1

    return registered
