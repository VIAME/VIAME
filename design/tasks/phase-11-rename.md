# Phase 11 (optional): rename to viame

Open decision 8 is answered: yes, and the `kwiver` shim lasts one release
-- the compat header, the `kwiver` python package and the `kwiver::` CMake
target names ship in the release phase 11 lands in and go in the next.

The rename is what unblocks dissolving `python/`. The bindings cannot sit
beside the C++ they bind while the package is still called `kwiver` and the
directory is `library/` -- 48 of the 67 files under `vital/types` share a
name with the C++ file they bind, so the package tree and the library tree
have to be the same tree before they can be merged. P11-T02b is that merge,
and it is the last thing that makes `library/<capability>/` the only place
VIAME's code lives.

### P11-T01 C++ namespace rename
Depends: P10-T06
Do:
- Script: `kwiver::vital` -> `viame`, `sprokit` -> `viame::pipeline`, `kwiver::arrows::*` -> `viame::*`; export macros renamed; `include/viame/compat/` provides old-namespace aliases (`namespace kwiver { namespace vital = ::viame; }`) for one release.
Done when:
- Build; BASELINE; out-of-tree example builds with and without the compat header.

### P11-T02 Python module rename
Depends: P11-T01
Do:
- `kwiver.vital.types` -> `viame.types`, `kwiver.vital.algo` -> `viame.algo`, `kwiver.sprokit.pipeline` -> `viame.pipeline`; `python/kwiver/` shim package re-exporting; all in-tree imports updated by script.
Done when:
- pytest passes with the shim; passes again with the shim removed from PYTHONPATH (in-tree code no longer needs it).

### P11-T02b Dissolve `python/`
Depends: P11-T02
Do:
- With the package renamed there is nothing keeping the bindings in a tree of their own: `python/kwiver/vital/types/*.py` moves to `library/core_types/`, `python/kwiver/vital/{config,util,applets,exceptions,io,modules,plugin_management,plugins}` to `library/algorithm_framework/` under the matching subdirectory, `python/kwiver/sprokit/{pipeline,pipeline_util,adapters,schedulers,processes}` to `library/pipeline_framework/`, `python/kwiver/arrows/*` to the library that owns each arrow, `python/kwiver/tools` to `tools/`. The binding for a C++ file sits beside it, which is the rule P2-T01 already applies to every other library's python.
- `viame_add_python_package` already globs a directory, so each library's `CMakeLists.txt` gains nothing but the package name; `python/kwiver/*/CMakeLists.txt` go away.
- `python/kwiver/{vital,sprokit}/tests` follow the code they test into `library/<dir>/tests/`, the way P2-T09 relocated `tests/plugins/*`.
- What is left of `python/` is packaging, not code: `requirements/` (the lock files P1-T08 wrote), `patches/`, `setup.py`, `pyproject.toml`. Those move to `packaging/` at the root; nothing imports them.
Done when:
- `python/` does not exist; `git grep -l "python/kwiver"` empty; pytest passes; BASELINE passes.

### P11-T03 Env var and log-level rename
Depends: P11-T02
Do:
- `VIAME_LOG_LEVEL`, `VIAME_PLUGIN_PATH` primary; old names read with a one-time warning.
Done when:
- DIVE smoke passes; release notes updated.
