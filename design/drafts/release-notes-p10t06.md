# Draft: release notes and migration guide for the renames

P10-T06. **Not applied to `RELEASE_NOTES.md`** -- that file is the user's to
edit. This is the text for it, to take, change or discard.

---

v0.24.0 (unreleased)
====================


- One C++ library and one python package: `libviame` and `viame`, replacing the `kwiver::vital*` libraries and the `kwiver` python package


- The names from before the rename keep working for one release: `kwiver::vital_algo` still links, `kwiver.vital.types` still imports, and `KwiverProcess` is still a base class


- `VIAME_LOG_LEVEL`, `VIAME_PLUGIN_PATH` and six more environment variables replace their `KWIVER_` and `SPROKIT_` names, which are still read and say so once


- 159 registered algorithms, processes, applets and schedulers were removed on purpose; `tests/baseline/removed.json` records each one with the reason


- 18 applets were renamed upstream into `viame` subcommands: `check-gpu` is `viame gpu`, `plot-detections` is `viame plot detections`, and so on



Migrating from v0.23.x
----------------------

**A shipped pipeline or an add-on needs no change.** No `.pipe` or `.conf`
file in `configs/`, and no add-on, names any of the renamed things: they name
algorithms and processes, and those names did not change. An add-on that is
pipelines, configs and model files is unaffected.

What changed is the surface a plugin written **outside** VIAME compiles and
imports against.

### C++ plugins

Include paths changed and are **not** aliased:

    <vital/types/image.h>            ->  <viame/core_types/image.h>
    <vital/config/config_block.h>    ->  <viame/algorithm_framework/config/config_block.h>
    <sprokit/pipeline/process.h>     ->  <viame/pipeline_framework/process.h>

If an existing `include/vital` or `include/sprokit` directory is still in
your install prefix, it is left over from an older install -- `make install`
only ever adds -- and a clean install of this release will not have it. Code
that still includes those paths compiles today and stops compiling on a fresh
prefix, so change the includes rather than relying on what is sitting there.

Namespaces are aliased, for one release. `viame::` is the name now, and one
header keeps the old spellings compiling:

    #include <viame/compat/kwiver.h>

    kwiver::vital::image_container_sptr image;   // viame::image_container_sptr
    sprokit::process_t proc;                     // viame::pipeline::process_t

It aliases namespaces and nothing else: it will not fix an `#include` line.

Link either name. `viame::viame_algo` and the rest are what the libraries are
called now; `kwiver::vital`, `kwiver::vital_algo`, `kwiver::vital_config`,
`kwiver::vital_logger`, `kwiver::vital_exceptions`, `kwiver::vital_util`,
`kwiver::vital_vpm`, `kwiver::vital_types`, `kwiver::vital_applets`,
`kwiver::sprokit_pipeline`, `kwiver::sprokit_pipeline_util` and
`kwiver::kwiver_adapter` are kept for one release, and each resolves to the
single library.

### Python plugins

The package is `viame`. Old module names import for one release, resolving to
the same module object -- a class imported by either name is one class, so
`isinstance` and plugin registration see one type:

    kwiver.vital.types      ->  viame.types
    kwiver.vital.algo       ->  viame.algo
    kwiver.vital.config     ->  viame.config
    kwiver.sprokit.pipeline ->  viame.pipeline
    kwiver.sprokit.processes->  viame.processes
    kwiver.tools            ->  viame.tools

`library/compat/kwiver/aliases.py` holds all 24. Two names changed spelling
rather than path: `KwiverProcess` is `ViameProcess` in `viame.processes.base`
(the old name is still exported from there), and `viame.util.VitalPIL` is
`viame.util.pil`.

### Environment variables

Each new name is read first; the old one is still read and warns once.

    VIAME_LOG_LEVEL           <-  KWIVER_DEFAULT_LOG_LEVEL
    VIAME_PLUGIN_PATH         <-  KWIVER_PLUGIN_PATH
    VIAME_PIPE_INCLUDE_PATH   <-  SPROKIT_PIPE_INCLUDE_PATH
    VIAME_NO_PYTHON_MODULES   <-  SPROKIT_NO_PYTHON_MODULES
    VIAME_PIPELINE_RUNNER     <-  SPROKIT_PIPELINE_RUNNER
    VIAME_PYTHON_PLUGIN_PATH  <-  KWIVER_PYTHON_PLUGIN_PATH
    VIAME_PYTHON_LOG_LEVEL    <-  KWIVER_PYTHON_DEFAULT_LOG_LEVEL
    VIAME_PYTHON_COLOREDLOGS  <-  KWIVER_PYTHON_COLOREDLOGS

`VIAME_PLUGIN_PATH` names plugin **libraries**, not directories: the loader
that scanned a directory is gone, so `KWIVER_PLUGIN_PATH` is read only to say
that it does nothing.

### Removed names

`tests/baseline/removed.json` lists all 159 with a reason each. Most went
because nothing selected them: no shipped pipeline or config named them and
no VIAME code did either. If a pipeline of yours names one, the reason field
says what happened to it and what replaced it where anything did.

`examples/plugin_creation/` builds against the install and is the worked
example for both languages.


