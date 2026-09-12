# This file is part of VIAME, and is distributed under an OSI-approved        #
# BSD 3-Clause License. See top-level LICENSE.txt or                          #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.           #
"""Import every module the install has to be able to import.

This exists because of a class of failure a build cannot otherwise catch: a
native dependency that links but cannot load. The motivating case was a
sqlite3.dll built from an export list older than its own source, which left
``_sqlite3.pyd`` unable to resolve ``sqlite3_close_v2``. Nothing referenced
sqlite3 during the build, so the install packaged cleanly and passed a detector
smoke test -- then every RF-DETR training run died hours in, inside
``ensure_rfdetr_compatibility()``, because ``transformers`` imports
``huggingface_hub``, which imports ``sqlite3``.

An import is the cheapest possible assertion that a shipped extension module
actually loads. Each module here is one test, so a failure names the culprit
rather than the first thing that tripped over it.
"""
import importlib

import pytest


# Standard-library modules that pull in a shipped native extension. These are
# the ones that fail when a bundled DLL is wrong, and the ones nothing else in
# the test suite would touch.
STDLIB_NATIVE = [
    "bz2",
    "ctypes",
    "decimal",
    "hashlib",
    "lzma",
    "sqlite3",
    "ssl",
    "zlib",
]

# Third-party packages VIAME's own plugins import at module scope, so a broken
# one takes out the plugin that needs it.
THIRD_PARTY = [
    "cv2",
    "huggingface_hub",
    "kwcoco",
    "kwimage",
    "numpy",
    "onnxruntime",
    "PIL",
    "pytorch_lightning",
    "scipy",
    "shapely",
    "torch",
    "torchvision",
    "transformers",
]

# VIAME and KWIVER packages. viame.pytorch pulls the trainers' shared
# utilities, which is where the RF-DETR and netharn entry points live.
VIAME = [
    "kwiver.vital.algo",
    "viame.core",
    "viame.onnx",
    "viame.pytorch",
    "viame.pytorch.utilities",
]


def _import(name):
    try:
        importlib.import_module(name)
    except ImportError as ex:
        # An optional component that was not built is a skip; a component that
        # was built but cannot load is a failure. The distinction is the
        # message: a missing module says so, a bad DLL does not.
        text = str(ex).lower()
        if "no module named" in text:
            pytest.skip(f"{name} is not part of this build: {ex}")
        pytest.fail(f"{name} is installed but does not import: {ex}")


@pytest.mark.parametrize("name", STDLIB_NATIVE)
def test_stdlib_native_imports(name):
    """A stdlib module backed by a bundled native library must load."""
    # Deliberately not skippable: these ship with every build, so "no module
    # named" here is itself a packaging failure.
    importlib.import_module(name)


@pytest.mark.parametrize("name", THIRD_PARTY)
def test_third_party_imports(name):
    _import(name)


@pytest.mark.parametrize("name", VIAME)
def test_viame_imports(name):
    _import(name)


def test_sqlite3_is_usable():
    """sqlite3 imported; check the entry points it actually calls.

    The failure this guards against was a missing DLL export, which surfaces on
    use rather than on import for anything resolved lazily.
    """
    import sqlite3

    con = sqlite3.connect(":memory:")
    try:
        con.execute("create table t (a integer, b text)")
        con.executemany("insert into t values (?, ?)",
                        [(1, "one"), (2, "two")])
        assert con.execute("select count(*) from t").fetchone()[0] == 2
        # sqlite3_expanded_sql / sqlite3_trace_v2, both absent from the stale
        # export list, back set_trace_callback.
        con.set_trace_callback(None)
    finally:
        con.close()


def test_torch_reports_its_build_config():
    """torch loads its native library and can be asked about itself."""
    import torch

    assert torch.__version__
    assert isinstance(torch.__config__.show(), str)
