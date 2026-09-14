# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Old module paths that resolve to where the module lives now.

Phase 2 moved each plugin's python into the libraries in `library/`, so
`viame.pytorch.netharn`, `viame.core.interactive_service` and the like became
`viame.object_detectors.netharn.netharn`, `viame.segmentation.
interactive_service`. Pipelines, add-ons, DIVE and people's own scripts still
name the old paths. `install()` makes each old name import the module at its
new one:

* the **same module object** is returned, so a class imported by either name
  is one class, and `isinstance` and registration see one type;
* `python -m old.path` runs the new module's code, as `runpy` asks the loader
  for it;
* an aliased package's submodules resolve through the alias as well --
  `viame.pytorch.netharn.data.x` -- and never through the ordinary path
  finder, which would load a second copy under the old name.

Nothing is imported until an old name is.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

__all__ = ["install"]


class _AliasLoader(importlib.abc.Loader):

    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        # `create_module` handed back the real, already executed module
        pass

    # what `runpy` needs for `python -m alias`
    def _target_spec(self):
        spec = importlib.util.find_spec(self.target)
        if spec is None or spec.loader is None:
            raise ImportError("no module named {!r}".format(self.target))
        return spec

    def get_code(self, fullname):
        spec = self._target_spec()
        if spec.submodule_search_locations is not None:
            # `python -m package` runs `package.__main__`
            main = importlib.util.find_spec(self.target + ".__main__")
            if main is None:
                raise ImportError("{!r} is a package and cannot be executed "
                                  "directly".format(self.target))
            return main.loader.get_code(self.target + ".__main__")
        return spec.loader.get_code(self.target)

    def is_package(self, fullname):
        return self._target_spec().submodule_search_locations is not None

    def get_filename(self, fullname):
        return self._target_spec().origin


class _AliasFinder(importlib.abc.MetaPathFinder):

    def __init__(self):
        self.aliases = {}

    def resolve(self, fullname):
        """The new name for `fullname`, longest matching alias first."""
        name = fullname
        suffix = ""
        while name:
            target = self.aliases.get(name)
            if target is not None:
                return target + suffix
            head, _, tail = name.rpartition(".")
            suffix = "." + tail + suffix
            name = head
        return None

    def find_spec(self, fullname, path=None, target=None):
        new = self.resolve(fullname)
        if new is None or new == fullname:
            return None
        loader = _AliasLoader(new)
        target = loader._target_spec()
        # The moved module's location, so that `python -m old.path` sets
        # `sys.argv[0]` and `__file__` from a real file, as `-m new.path`
        # does -- `runpy` takes both from the spec, and argparse, among
        # others, builds its program name from `sys.argv[0]`
        spec = importlib.machinery.ModuleSpec(
            fullname, loader, origin=target.origin,
            is_package=target.submodule_search_locations is not None)
        spec.has_location = target.has_location
        return spec


_FINDER = _AliasFinder()


def install(aliases):
    """Make each old name in `aliases` import its new one.

    `aliases` maps an old module or package name to the new one. A package
    entry covers everything below it.
    """
    _FINDER.aliases.update(aliases)
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
