#!/usr/bin/env python3
"""Rename the python package from `kwiver.*` to `viame.*`.

P11-T02. The C++ namespaces became `viame` in P11-T01; this is the same
rename one layer up, and it is what lets P11-T02b dissolve `python/` -- a
binding cannot sit beside the C++ it binds while its package says `kwiver`.

What this rewrites:

* dotted module names, in python imports and in the 121 C++
  `py::module::import( "kwiver.vital.config" )` strings that a python-side
  rename would not touch;
* the entry point group names, which are module-shaped and are the one part
  of this that code outside VIAME writes;
* two identifiers that carry the old name in their spelling rather than in a
  package path: `KwiverProcess`, the base class every python process derives
  from, and `vital_logging`.

What it deliberately does not rewrite:

* **CMake module paths.** `python/kwiver/*/CMakeLists.txt` is deleted by
  P11-T02b rather than edited, and the two `python.cmake` files that already
  live beside their C++ are changed by hand. The target rename learned this
  the hard way: a script that rewrites CMake arguments cannot tell a module
  path from a directory name, and silently installed a package into
  `site-packages/kwiver/viame_algorithm_framework/`.
* **A bare `kwiver`.** There is no catch-all entry, because the word is also
  a URL, a repository, a plugin name (`kwiver_processes_adapters`) and a
  directory. Only the names listed below move; anything left over is
  reported rather than guessed at.

Run `--check` to list what would change without changing it.
"""

import argparse
import os
import re
import subprocess
import sys


# Old module name -> new module name. Applied longest first, so that
# `kwiver.vital.types` is taken before `kwiver.vital` and `.uid` survives on
# the end of it.
MODULES = (
    # The process base class moves with its module: `kwiver_process` was the
    # file, `KwiverProcess` the class, and `base` says what it is.
    ( "kwiver.sprokit.processes.kwiver_process", "viame.processes.base" ),

    ( "kwiver.vital.plugin_management", "viame.plugin_management" ),
    ( "kwiver.vital.test_interface", "viame.test_interface" ),
    ( "kwiver.vital.vital_logging", "viame.log" ),
    ( "kwiver.vital.exceptions", "viame.exceptions" ),
    ( "kwiver.vital.applets", "viame.applets" ),
    ( "kwiver.vital.modules", "viame.modules" ),
    ( "kwiver.vital.plugins", "viame.plugins" ),
    ( "kwiver.vital.config", "viame.config" ),
    ( "kwiver.vital.tests", "viame.tests" ),
    ( "kwiver.vital.types", "viame.types" ),
    ( "kwiver.vital.algo", "viame.algo" ),
    ( "kwiver.vital.util", "viame.util" ),
    ( "kwiver.vital.io", "viame.io" ),
    ( "kwiver.vital", "viame" ),

    # `sprokit` was two levels deep and is one now, as the C++ namespace is.
    # `pipeline_util` is the python side of `viame_pipeline_util`, so it
    # lands under the pipeline package rather than beside it.
    ( "kwiver.sprokit.util.test", "viame.pipeline.tests.util" ),
    ( "kwiver.sprokit.pipeline_util", "viame.pipeline.util" ),
    ( "kwiver.sprokit.schedulers", "viame.schedulers" ),
    ( "kwiver.sprokit.processes", "viame.processes" ),
    ( "kwiver.sprokit.adapters", "viame.adapters" ),
    ( "kwiver.sprokit.pipeline", "viame.pipeline" ),
    ( "kwiver.sprokit.tests", "viame.pipeline.tests" ),
    ( "kwiver.sprokit", "viame.pipeline" ),

    ( "kwiver.tools", "viame.tools" ),

    # Entry point groups. These are what a package outside VIAME writes in
    # its own `pyproject.toml`, so `discovery.py` reads both names for one
    # release -- see the shim it installs.
    ( "kwiver.python_plugin_registration", "viame.python_plugin_registration" ),
    ( "kwiver.cpp_search_paths", "viame.cpp_search_paths" ),
    ( "kwiver.python_plugins", "viame.python_plugins" ),
)

# Names whose spelling carries the old package rather than a path to it.
# `vital_logging` has no collision to worry about: none of the six modules
# that import it uses the name `log` for anything else.
TOKENS = (
    ( "KwiverProcess", "ViameProcess" ),
    ( "vital_logging", "log" ),
)

# `from kwiver import X` and `import kwiver`: the top level package by
# itself, which the dotted table cannot express.
PATTERNS = (
    ( re.compile( r"(?<![\w.])from kwiver import\b" ), "from viame import" ),
    ( re.compile( r"(?<![\w.])import kwiver(?![\w.])" ), "import viame" ),
)

SUFFIXES = (
    ".py", ".pyi", ".cxx", ".h", ".txx", ".cmake", ".txt", ".rst", ".md",
    ".in", ".toml", ".sh", ".pipe", ".conf", ".yml",
)

# `design/` is the plan and records the old names on purpose. `packages/` and
# `library/tpl/` are other people's code. `tests/baseline/` is re-recorded
# from a run rather than edited, so that the baseline stays a measurement.
SKIP_PREFIXES = (
    "design/",
    "packages/",
    "library/tpl/",
    "tests/baseline/",
)

# What a finished run should not leave behind. A `kwiver.` that is a URL
# (`kwiver.readthedocs.io`) is not module-shaped and does not match.
RESIDUE = re.compile(
    r"(?<![\w.])kwiver\.(?:vital|sprokit|arrows|tools|python_plugin"
    r"|python_plugins|cpp_search_paths)[\w.]*" )


def _by_length( pairs ):
    """Longest source first, so a prefix never eats a longer name."""
    return tuple( sorted( pairs, key = lambda p: -len( p[ 0 ] ) ) )


def rewrite( text ):
    """Return `(new_text, changes)` for one file's contents."""
    changes = 0

    for old, new in _by_length( MODULES ):
        # Not preceded by a word character or a dot, so that a longer dotted
        # name that was already rewritten is not rewritten again, and not
        # followed by a word character, so `kwiver.vital.io` does not match
        # inside `kwiver.vital.iodine`. A trailing `.` is fine and common:
        # `kwiver.vital.types.uid`.
        pattern = re.compile(
            r"(?<![\w.])" + re.escape( old ) + r"(?![\w])" )
        text, n = pattern.subn( new, text )
        changes += n

    for old, new in TOKENS:
        pattern = re.compile( r"(?<![\w.])" + re.escape( old ) + r"\b" )
        text, n = pattern.subn( new, text )
        changes += n

    for pattern, new in PATTERNS:
        text, n = pattern.subn( new, text )
        changes += n

    return text, changes


def tracked_files( root ):
    """Every tracked file this should look at, as repository paths."""
    out = subprocess.run(
        [ "git", "ls-files" ], cwd = root, check = True,
        stdout = subprocess.PIPE, text = True ).stdout.split( "\n" )

    for path in out:
        if not path:
            continue
        if path.startswith( SKIP_PREFIXES ):
            continue
        if not path.endswith( SUFFIXES ):
            continue
        yield path


def main( argv = None ):
    parser = argparse.ArgumentParser( description = __doc__ )
    parser.add_argument( "--root", default = ".",
                         help = "the repository to rewrite" )
    parser.add_argument( "--check", action = "store_true",
                         help = "report what would change, change nothing" )
    args = parser.parse_args( argv )

    touched = 0
    changes = 0
    residue = []

    for path in tracked_files( args.root ):
        full = os.path.join( args.root, path )
        try:
            with open( full, encoding = "utf-8" ) as handle:
                before = handle.read()
        except UnicodeDecodeError:
            continue

        after, n = rewrite( before )
        if n:
            touched += 1
            changes += n
            if args.check:
                print( "{:4d}  {}".format( n, path ) )
            else:
                with open( full, "w", encoding = "utf-8" ) as handle:
                    handle.write( after )

        for match in RESIDUE.finditer( after ):
            residue.append( "{}: {}".format( path, match.group( 0 ) ) )

    print( "\n{} name(s) in {} file(s){}".format(
        changes, touched, " (--check: nothing written)" if args.check else "" ) )

    if residue:
        print( "\n{} old name(s) left, for a person to look at:".format(
            len( residue ) ) )
        for line in residue[ :40 ]:
            print( "  " + line )
        if len( residue ) > 40:
            print( "  ... and {} more".format( len( residue ) - 40 ) )

    return 0


if __name__ == "__main__":
    sys.exit( main() )
