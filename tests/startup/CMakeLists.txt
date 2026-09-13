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

set_property( TEST "startup:budget" PROPERTY LABELS BASELINE STARTUP )
set_property( TEST "startup:imports" PROPERTY LABELS BASELINE STARTUP )
set_property( TEST "startup:budget" PROPERTY TIMEOUT 300 )
set_property( TEST "startup:imports" PROPERTY TIMEOUT 300 )
set_property( TEST "startup:budget" PROPERTY RUN_SERIAL TRUE )
set_property( TEST "startup:imports" PROPERTY RUN_SERIAL TRUE )
