"""
SMQTK Plugin utilities - Minimal port for VIAME search functionality.
"""
import abc
import importlib
import os

import six

from . import Configurable, merge_dict


@six.add_metaclass(abc.ABCMeta)
class Pluggable(Configurable):
    """
    Base class for pluggable SMQTK components.
    """

    @classmethod
    @abc.abstractmethod
    def is_usable(cls):
        """
        Check whether this class is available for use.
        """
        raise NotImplementedError()


def make_config(plugin_impls):
    """
    Create a default configuration dictionary for a set of plugin implementations.

    :param plugin_impls: Dictionary mapping implementation names to class types.
    :return: Configuration dictionary with 'type' key and sub-configs.
    """
    d = {'type': None}
    for label, cls in plugin_impls.items():
        if hasattr(cls, 'is_usable') and cls.is_usable():
            d[label] = cls.get_default_config()
    return d


def to_plugin_config(inst):
    """
    Create a plugin configuration dictionary from an instance.

    :param inst: Instance to get configuration from.
    :return: Configuration dictionary.
    """
    type_name = inst.__class__.__name__
    return {
        'type': type_name,
        type_name: inst.get_config()
    }


def from_plugin_config(config, plugin_impls, *args):
    """
    Instantiate a plugin class from a configuration dictionary.

    :param config: Configuration dictionary with 'type' key.
    :param plugin_impls: Dictionary mapping implementation names to class types.
    :param args: Additional positional arguments to pass to from_config.
    :return: New instance of the specified plugin type.
    """
    t = config.get('type')
    if t is None:
        raise ValueError("No 'type' specified in configuration.")

    cls = plugin_impls.get(t)
    if cls is None:
        raise ValueError("Unknown plugin type: %s" % t)

    cls_config = config.get(t, {})
    return cls.from_config(cls_config, *args)


def get_plugins(package_name, package_dir, env_var, helper_var, base_class,
                reload_modules=False):
    """
    Discover and return plugin implementations.

    This is a simplified version that discovers classes in sibling modules.

    :param package_name: Name of the package to search.
    :param package_dir: Directory path of the package.
    :param env_var: Environment variable containing additional module paths.
    :param helper_var: Variable name in modules that may specify exported classes.
    :param base_class: Base class that plugins should inherit from.
    :param reload_modules: Whether to reload modules.
    :return: Dictionary mapping class names to class types.
    """
    impls = {}

    # Search for modules in the package directory
    if os.path.isdir(package_dir):
        for fname in os.listdir(package_dir):
            if fname.startswith('_') or not fname.endswith('.py'):
                continue

            module_name = fname[:-3]
            full_module_name = package_name + '.' + module_name

            try:
                if reload_modules:
                    module = importlib.import_module(full_module_name)
                    importlib.reload(module)
                else:
                    module = importlib.import_module(full_module_name)
            except Exception:
                continue

            # Check for helper variable
            if hasattr(module, helper_var):
                helper = getattr(module, helper_var)
                if helper is None:
                    continue
                if isinstance(helper, type) and issubclass(helper, base_class):
                    impls[helper.__name__] = helper
                elif hasattr(helper, '__iter__'):
                    for cls in helper:
                        if isinstance(cls, type) and issubclass(cls, base_class):
                            impls[cls.__name__] = cls
            else:
                # Look for classes that inherit from base_class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, base_class) \
                            and attr is not base_class:
                        impls[attr.__name__] = attr

    return impls


__all__ = [
    'Pluggable',
    'make_config',
    'to_plugin_config',
    'from_plugin_config',
    'get_plugins',
]
