cmake_minimum_required( VERSION 3.16 )

file( MAKE_DIRECTORY "${TEST_DIRECTORY}/unrelated" )
file( WRITE "${TEST_DIRECTORY}/check.sh" [=[
set -e
expected=$2
original_dir=$PWD
# A nonempty CDPATH must not pollute command-substitution output.
CDPATH="$PWD"
PS1="original prompt> "
if [ "$3" = function ]; then
  load_viame() { . "$1"; }
  load_viame "$1"
else
  . "$1"
fi
[ "$VIAME_INSTALL" = "$expected" ]
[ "$PWD" = "$original_dir" ]
[ "${PATH%%:*}" = "$expected/bin" ]
case "$PYTHONPATH" in "$expected/"*) ;; *) exit 1 ;; esac
case "$KWIVER_PLUGIN_PATH" in "$expected/"*) ;; *) exit 1 ;; esac
[ "${viame_setup_source+x}" != x ]
[ "$PS1" = "(viame) original prompt> " ]
. "$1"
[ "$VIAME_INSTALL" = "$expected" ]
[ "$PS1" = "(viame) original prompt> " ]
]=] )

# Exercise both platform branches and both distributed setup templates.
foreach( APPLE OFF ON )
  include( "${VIAME_SOURCE_DIR}/cmake/set_setup_script_vars.cmake" )
  foreach( template dev rel )
    set( install_dir "${TEST_DIRECTORY}/${template}-${APPLE} install [1]" )
    file( MAKE_DIRECTORY "${install_dir}" )
    get_filename_component( install_dir "${install_dir}" REALPATH )
    configure_file( "${VIAME_SOURCE_DIR}/cmake/setup_viame.${template}.sh.in"
      "${install_dir}/setup_viame.sh" @ONLY )
    set( link_dir "${TEST_DIRECTORY}/${template}-${APPLE}-link" )
    file( CREATE_LINK "${install_dir}" "${link_dir}" SYMBOLIC )
    file( RELATIVE_PATH relative_script "${TEST_DIRECTORY}/unrelated"
      "${install_dir}/setup_viame.sh" )
    foreach( script "${install_dir}/setup_viame.sh"
                    "${relative_script}" "${link_dir}/setup_viame.sh" )
      foreach( mode direct function )
        execute_process( COMMAND "${CMAKE_COMMAND}" -E env --unset=CUDA_INSTALL_DIR
          "${SHELL_EXECUTABLE}" -f "${TEST_DIRECTORY}/check.sh"
          "${script}" "${install_dir}" "${mode}"
          WORKING_DIRECTORY "${TEST_DIRECTORY}/unrelated"
          RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error )
        if( NOT result EQUAL 0 OR NOT "${output}${error}" STREQUAL "" )
          message( FATAL_ERROR
            "${SHELL_EXECUTABLE}: ${template}/${APPLE}/${script}/${mode}: "
            "${result}\n${output}${error}" )
        endif()
      endforeach()
    endforeach()
  endforeach()
endforeach()
message( STATUS "Setup sourcing checks passed for ${SHELL_EXECUTABLE}" )
