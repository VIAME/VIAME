# Compatibility baseline tests
#
# Each dumps what the install currently provides and compares it against the
# committed baseline. They run against the install prefix, so they need the
# setup script sourced the same way the other SOURCE_SETUP tests do.

if( NOT Python3_EXECUTABLE )
  find_package( Python3 COMPONENTS Interpreter REQUIRED )
endif()

set( baseline_dir "${CMAKE_CURRENT_SOURCE_DIR}" )
set( install_dir "${VIAME_BUILD_INSTALL_PREFIX}" )
set( output_dir "${CMAKE_CURRENT_BINARY_DIR}" )

if( WIN32 )
  set( baseline_setup "call \"${install_dir}/setup_viame.bat\"" )
else()
  set( baseline_setup "source \"${install_dir}/setup_viame.sh\"" )
endif()

add_test(
  NAME "baseline:registry"
  COMMAND bash -c
    "${baseline_setup} && \
     viame registry-dump --json --output '${output_dir}/registry.json' && \
     '${Python3_EXECUTABLE}' '${baseline_dir}/compare_registry.py' \
       '${baseline_dir}/registry.json' '${output_dir}/registry.json' \
       --removed '${baseline_dir}/removed.json' \
       --pending '${baseline_dir}/pending.json'"
)

add_test(
  NAME "baseline:pipes"
  COMMAND bash -c
    "${baseline_setup} && \
     viame pipe-check --all --json --output '${output_dir}/pipes.json' && \
     '${Python3_EXECUTABLE}' '${baseline_dir}/compare_pipes.py' \
       '${baseline_dir}/pipes.json' '${output_dir}/pipes.json' \
       --removed '${baseline_dir}/removed_pipes.json'"
)

set_property( TEST "baseline:registry" PROPERTY LABELS BASELINE )
set_property( TEST "baseline:pipes" PROPERTY LABELS BASELINE )
set_property( TEST "baseline:registry" PROPERTY TIMEOUT 900 )
set_property( TEST "baseline:pipes" PROPERTY TIMEOUT 1800 )
