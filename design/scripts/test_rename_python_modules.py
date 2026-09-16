"""Fixtures for design/scripts/rename_python_modules.py, before it runs.

Every case is taken from the tree, including the ones that must not change:
the target rename's six classes of over-reach all came from a script that
was never shown a case it should leave alone.
"""

import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location(
    "rp", os.path.join( os.path.dirname( os.path.abspath( __file__ ) ),
                        "rename_python_modules.py" ) )
rp = importlib.util.module_from_spec( spec )
spec.loader.exec_module( rp )

CASES = [
    (
        "the long name is taken before the short one",
        "from kwiver.vital.types import Image, ImageContainer\n",
        "from viame.types import Image, ImageContainer\n",
    ),
    (
        "a submodule survives on the end of a mapped name",
        'py::module::import( "kwiver.vital.types.uid" );\n',
        'py::module::import( "viame.types.uid" );\n',
    ),
    (
        "the middle name disappears, as it did in C++",
        "import kwiver.vital.algo as algo\n",
        "import viame.algo as algo\n",
    ),
    (
        "sprokit collapses to one level",
        "from kwiver.sprokit.pipeline import process, datum\n",
        "from viame.pipeline import process, datum\n",
    ),
    (
        "pipeline_util lands under the pipeline package",
        "from kwiver.sprokit.pipeline_util import bake\n",
        "from viame.pipeline.util import bake\n",
    ),
    (
        # The base class every python process in VIAME derives from: module
        # and class rename together.
        "the process base class and its module",
        "from kwiver.sprokit.processes.kwiver_process import KwiverProcess\n"
        "class ImageViewer( KwiverProcess ):\n"
        "    def __init__( self, conf ):\n"
        "        KwiverProcess.__init__( self, conf )\n",
        "from viame.processes.base import ViameProcess\n"
        "class ImageViewer( ViameProcess ):\n"
        "    def __init__( self, conf ):\n"
        "        ViameProcess.__init__( self, conf )\n",
    ),
    (
        "the logging helper loses the old name in its spelling too",
        "from kwiver.vital import vital_logging\n"
        "logger = vital_logging.getLogger( __name__ )\n",
        "from viame import log\n"
        "logger = log.getLogger( __name__ )\n",
    ),
    (
        "the bare top level package",
        "import kwiver\nfrom kwiver import PYTHON_PLUGIN_ENTRYPOINT\n",
        "import viame\nfrom viame import PYTHON_PLUGIN_ENTRYPOINT\n",
    ),
    (
        "entry point groups are module shaped and move too",
        'entry_points={ "kwiver.python_plugins": [\n'
        '    "say=kwiver.vital.test_interface.python_say" ] }\n',
        'entry_points={ "viame.python_plugins": [\n'
        '    "say=viame.test_interface.python_say" ] }\n',
    ),
    (
        "a string module name in a test monkeypatch",
        "sys.modules['kwiver.vital.types'] = types_stub\n",
        "sys.modules['viame.types'] = types_stub\n",
    ),
    (
        "the plugin package list",
        'BUILTIN_PLUGIN_PACKAGES = (\n'
        '    "kwiver.sprokit.processes",\n'
        '    "kwiver.sprokit.schedulers",\n'
        '    "viame.classifiers",\n'
        ')\n',
        'BUILTIN_PLUGIN_PACKAGES = (\n'
        '    "viame.processes",\n'
        '    "viame.schedulers",\n'
        '    "viame.classifiers",\n'
        ')\n',
    ),
    (
        "vital on its own is the root package now",
        "from kwiver.vital import ConfigBlock\n",
        "from viame import ConfigBlock\n",
    ),
    # ---------------------------------------------------------------- leave
    (
        # `kwiver_processes_adapters` is a C++ plugin name, not a module, and
        # it starts with the same letters as the class that does move.
        "a plugin name that merely starts the same way",
        'process_registrar reg( vpm, "kwiver_processes_adapters" );\n',
        'process_registrar reg( vpm, "kwiver_processes_adapters" );\n',
    ),
    (
        "a URL is not a module",
        "# KWIVER documentation: https://kwiver.readthedocs.io/\n"
        "# https://github.com/Kitware/kwiver/blob/master/LICENSE\n",
        "# KWIVER documentation: https://kwiver.readthedocs.io/\n"
        "# https://github.com/Kitware/kwiver/blob/master/LICENSE\n",
    ),
    (
        "a source path is not a module",
        "#include <python/kwiver/internal/python_plugin_factory.h>\n",
        "#include <python/kwiver/internal/python_plugin_factory.h>\n",
    ),
    (
        # P11-T01 already did the C++; a namespace must not be touched again.
        "the renamed C++ namespace stays renamed",
        "viame::config_block_sptr config;\nkv::plugin_manager::instance();\n",
        "viame::config_block_sptr config;\nkv::plugin_manager::instance();\n",
    ),
    (
        "an environment variable is P11-T03's, not this one's",
        'if "KWIVER_PYTHON_DEFAULT_LOG_LEVEL" in os.environ:\n',
        'if "KWIVER_PYTHON_DEFAULT_LOG_LEVEL" in os.environ:\n',
    ),
    (
        "a package that was already viame is left alone",
        "from viame.object_detectors.base import Detector\n",
        "from viame.object_detectors.base import Detector\n",
    ),
]

# Running twice must not change anything the first run produced.
IDEMPOTENT = [ case[ 2 ] for case in CASES ]


def main():
    failures = 0

    for name, before, expected in CASES:
        got, changed = rp.rewrite( before )
        if got != expected:
            failures += 1
            print( "FAIL: {}\n--- got ---\n{}--- expected ---\n{}".format(
                name, got, expected ) )
        else:
            print( "ok: {} ({} name{})".format(
                name, changed, "" if changed == 1 else "s" ) )

    for text in IDEMPOTENT:
        got, changed = rp.rewrite( text )
        if got != text or changed:
            failures += 1
            print( "FAIL: not idempotent\n{}\n->\n{}".format( text, got ) )
    print( "ok: a second run changes nothing" )

    print( "\n{} failure(s)".format( failures ) )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit( main() )
