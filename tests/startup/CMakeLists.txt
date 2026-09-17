# Startup budget
#
# Two claims P8-T10 makes, each checked the way it can actually fail.
#
# `startup:budget` is a clock. It is the only performance test in the tree,
# and it earns its place because the cost it guards came back twice while
# P8-T03 was being done: eager C++ registration took `viame help` from
# 0.04 s to 2.57 s, and a single module-level `import torch` in a python
# file that registers nothing cost 1.5 s on every command in VIAME.
#
# `startup:imports` is the same claim read from the import log, which says
# *what* was imported rather than how long it took. It fails identically on
# every machine; the clock does not.
#
# Both run alone: the budgets are wall-clock, and a dozen tests on the other
# cores turn a passing install into a failing one.

if( NOT Python3_EXECUTABLE )
  find_package( Python3 COMPONENTS Interpreter REQUIRED )
endif()

set( startup_dir "${CMAKE_CURRENT_SOURCE_DIR}" )
set( install_dir "${VIAME_BUILD_INSTALL_PREFIX}" )
set( output_dir "${CMAKE_CURRENT_BINARY_DIR}" )

if( WIN32 )
  set( startup_setup "call \"${install_dir}/setup_viame.bat\"" )
else()
  set( startup_setup "source \"${install_dir}/setup_viame.sh\"" )
endif()

# The task names `viame --version`; there is no such flag and no `version`
# applet, so the command that means what it meant -- dispatch answering
# without loading anything -- is `viame help`, which is measured instead.
#
# The dump writes to a file rather than the terminal so that the measurement
# is the dump and not the pipe.
add_test(
  NAME "startup:budget"
  COMMAND bash -c
    "${startup_setup} && \
     '${Python3_EXECUTABLE}' '${startup_dir}/budget.py' \
       --json '${output_dir}/startup.json' \
       'viame help=0.5:viame help' \
       'viame runner --help=0.5:viame runner --help' \
       'viame registry-dump=2.0:viame registry-dump --json --output \
         ${output_dir}/registry.json'"
)

add_test(
  NAME "startup:imports"
  COMMAND bash -c
    "${startup_setup} && \
     '${Python3_EXECUTABLE}' '${startup_dir}/imports.py' \
       --command 'viame runner --help' && \
     '${Python3_EXECUTABLE}' '${startup_dir}/imports.py' \
       --forbid torch mmdet \
       --command 'viame registry-dump --json --output \
         ${output_dir}/registry_imports.json'"
)

# What sourcing the setup script leaves behind. P10-T01's done-when, kept as
# a test because the variables it forbids are exactly the ones an edit
# copied from an older script would bring back: the plugin search paths
# (`KWIVER_PLUGIN_PATH*`) and the sprokit ones, none of which anything reads.
# The script is sourced into an environment with both families cleared first,
# so what is found came from the script and not from the shell that ran ctest.
# No `;` anywhere in the script: `add_test` would split the argument there.
if( NOT WIN32 )
  add_test(
    NAME "startup:environment"
    COMMAND env -u KWIVER_PLUGIN_PATH -u KWIVER_PLUGIN_PATH_NO_DEFAULTS
      -u SPROKIT_PIPE_INCLUDE_PATH -u SPROKIT_MODULE_PATH
      -u SPROKIT_PYTHON_MODULES -u VIAME_LOG_LEVEL
      -u VIAME_PIPE_INCLUDE_PATH -u VIAME_NO_PYTHON_MODULES
      -u VIAME_PYTHON_PLUGIN_PATH -u VIAME_PYTHON_LOG_LEVEL
      bash -c
      "${startup_setup} && \
       found=$(env | grep -E '^(SPROKIT|KWIVER_PLUGIN)' || true) && \
       ( test -z \"$found\" || ( echo \"left behind: $found\" && false ) ) && \
       test -n \"$VIAME_LOG_LEVEL\" && \
       test \"$VIAME_INSTALL\" -ef '${install_dir}'"
  )
  set_property( TEST "startup:environment" PROPERTY LABELS BASELINE STARTUP )
endif()

set_property( TEST "startup:budget" PROPERTY LABELS BASELINE STARTUP )
set_property( TEST "startup:imports" PROPERTY LABELS BASELINE STARTUP )
set_property( TEST "startup:budget" PROPERTY TIMEOUT 300 )
set_property( TEST "startup:imports" PROPERTY TIMEOUT 300 )
set_property( TEST "startup:budget" PROPERTY RUN_SERIAL TRUE )
set_property( TEST "startup:imports" PROPERTY RUN_SERIAL TRUE )
