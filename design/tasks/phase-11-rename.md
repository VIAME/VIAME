# Phase 11 (optional): rename to viame

Requires open decision 8 = yes.

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

### P11-T03 Env var and log-level rename
Depends: P11-T02
Do:
- `VIAME_LOG_LEVEL`, `VIAME_PLUGIN_PATH` primary; old names read with a one-time warning.
Done when:
- DIVE smoke passes; release notes updated.
