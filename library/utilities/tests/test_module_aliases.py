# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The old `viame.<plugin>` module paths.

Phase 2 dissolved `plugins/`, and `viame.core`, `viame.pytorch`,
`viame.opencv`, `viame.onnx` and `viame.colmap` stayed as packages of
aliases generated from `design/lite-file-map.tsv`. An alias whose target does
not exist is an old path that fails with a confusing error, so every one is
checked against the source tree; and an old name must import the very module
the new one does, or a class imported both ways is two classes.
"""

import importlib
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PACKAGES = ("core", "pytorch", "opencv", "onnx", "colmap")


def aliases(package):
    return importlib.import_module("viame.%s.aliases" % package).ALIASES


def source_exists(module):
    parts = module.split(".")[1:]
    if parts[-1].startswith("_"):
        return True  # a compiled extension; its sources are C++
    base = os.path.join(ROOT, "library", *parts)
    return os.path.exists(base + ".py") or os.path.isdir(base)


@pytest.mark.parametrize("package", PACKAGES)
def test_every_alias_names_a_module_in_the_tree(package):
    missing = [(old, new) for old, new in sorted(aliases(package).items())
               if not source_exists(new)]
    assert not missing, missing


def test_old_name_is_the_same_module_object():
    old = importlib.import_module("viame.core.interactive_service")
    new = importlib.import_module("viame.segmentation.interactive_service")
    assert old is new


def test_deleted_modules_are_not_aliased():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("viame.opencv.stereo_demo")


@pytest.mark.parametrize("package", PACKAGES)
def test_compatibility_package_declares_nothing(package):
    module = importlib.import_module("viame." + package)
    assert module.__vital_algorithm_declarations__ == []
    assert module.__sprokit_process_declarations__ == []
