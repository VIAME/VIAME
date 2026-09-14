# Finding python and working out where its packages go
#
# Kwiver's `kwiver-setup-python.cmake`, without the parts VIAME's build does
# not reach: the `SKBUILD` component list, `sprokit_python_output_path` (the
# sprokit configurations it existed for are gone), and the status block that
# printed eleven variables at every configure.
#
# The variable names are still kwiver's. The top-level CMakeLists overrides
# three of them straight after including this -- `kwiver_python_subdir`,
# `kwiver_python_output_path` and `kwiver_python_install_path`, because
# VIAME's install layout differs from kwiver's -- and renaming them is a
# separate change from replacing the file that computes them.

include_guard( GLOBAL )

#+
# Run a python snippet and capture its output, or stop the configure.
#-
function( _viame_pycmd outvar cmd )
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "${cmd}"
    RESULT_VARIABLE exitcode
    OUTPUT_VARIABLE output
    ${ARGN}
    )
  if( NOT ${exitcode} EQUAL 0 )
    message( FATAL_ERROR
      "Python command failed with code ${exitcode}:\n${cmd}" )
  endif()
  string( STRIP "${output}" output )
  set( ${outvar} "${output}" PARENT_SCOPE )
endfunction()

#+
# Stop the configure if a python package the build needs is not importable.
#-
function( _viame_require_pypackage package )
  # The install prefix's site-packages goes on the path too: a superbuild may
  # have pip-installed the package there rather than into the interpreter's
  # own.
  if( kwiver_python_install_path )
    set( command
      "import sys; sys.path.insert(0, '${kwiver_python_install_path}'); import ${package}" )
  else()
    set( command "import ${package}" )
  endif()

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "${command}"
    RESULT_VARIABLE exitcode
    OUTPUT_QUIET ERROR_QUIET
    )
  if( NOT ${exitcode} EQUAL 0 )
    message( FATAL_ERROR
      "${package} is missing. Install it in the python environment this "
      "build uses." )
  endif()
endfunction()

# ----------------------------------------------------------------------------
find_package( Python 3.8 REQUIRED
  COMPONENTS Interpreter Development.Module Development.Embed )

_viame_pycmd( PYTHON_VERSION
  "import sys, re; print(re.match(r'^[0-9]+\\.[0-9]+', sys.version)[0])" )
set( KWIVER_PYTHON_VERSION "${PYTHON_VERSION}" CACHE STRING "" )
mark_as_advanced( KWIVER_PYTHON_VERSION )

# Where this interpreter keeps installed packages, relative to its prefix. It
# varies by platform and by distribution, so it is asked rather than assumed.
_viame_pycmd( python_site_packages [==[
import sysconfig, os
base_path = sysconfig.get_config_var("base")
purelib_path = sysconfig.get_path("purelib", vars={"base": base_path})
# Force site-packages instead of dist-packages
purelib_path = purelib_path.replace("dist-packages", "site-packages")
rel_path = os.path.relpath(purelib_path, base_path)
# relpath uses the native separator, so this comes back with backslashes on
# Windows. They end up embedded in generated install scripts, where CMake
# reads them as escapes and stops with "Invalid character escape".
rel_path = rel_path.replace(os.sep, "/")
# Remove local/ prefix if present
if rel_path.startswith("local/"):
    rel_path = rel_path[6:]
print(rel_path)
]==] )

# Just the last component: "site-packages".
get_filename_component( python_sitename "${python_site_packages}" NAME )

# PEP 3149 ABI tag, for the extension module suffix.
_viame_pycmd( _abiflags
  "import sysconfig; print(sysconfig.get_config_var('ABIFLAGS'))" )
set( PYTHON_ABIFLAGS "${_abiflags}"
     CACHE STRING "The ABI flags for the version of Python being used" )
mark_as_advanced( PYTHON_ABIFLAGS )

# pybind11 is vendored in `library/tpl/pybind11`, which defines
# `pybind11::pybind11`, `::module` and `::embed`. It used to come from a
# `find_package` resolved through fletch.
if( NOT TARGET pybind11::pybind11 )
  message( FATAL_ERROR "library/tpl/pybind11 has not been added yet" )
endif()

set( kwiver_python_install_path
     "${CMAKE_INSTALL_PREFIX}/${python_site_packages}" )

# E.g. "lib/python3.10" then "python3.10".
get_filename_component( python_lib_subdir "${python_site_packages}" DIRECTORY )
get_filename_component( python_subdir "${python_lib_subdir}" NAME )
set( kwiver_python_subdir "${python_subdir}" )
set( kwiver_python_output_path "${KWIVER_BINARY_DIR}/${python_lib_subdir}" )

if( KWIVER_ENABLE_TESTS )
  _viame_require_pypackage( "pytest" )
endif()
