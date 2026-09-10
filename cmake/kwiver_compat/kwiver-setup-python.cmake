###
# Finds the python binaries, libs, include, and site-packages paths
#
# The purpose of this file is to export variables that will be used in
# kwiver/CMake/utils/kwiver-utils-python.cmake and
# kwiver/sprokit/conf/sprokit-macro-python.cmake (the latter will eventually be
# consolidated into the former)
#
# Exported variables used by python utility functions are:
#
#    PYTHON_VERSION
#      the major/minor python version
#
#    PYTHON_ABI_FLAGS
#      Python abstract binary interface flags (used internally for defining
#      subsequent variables, but settable by the user as an advanced setting)
#
#    python_site_packages
#      Location where python packages are installed relative to your python
#      install directory. For example:
#        Windows system install: Lib\site-packages
#        Debian system install: lib/python2.7/dist-packages
#        Debian virtualenv install: lib/python3.5/site-packages
#
#    python_sitename
#      The basename of the python_site_packages directory. This is either
#      site-packages (in most cases) or dist-packages (if your python was
#      configured by a debian package manager). If you are using a python
#      virtualenv (you should be) then this will be site-packages
#
#    kwiver_python_subdir
#      basename of the python lib folder (that contains site-packages).
#      Depends on the python major/minor version and the ABI flags
#      (e.g. python2.7, python3.5m)
#
#    kwiver_python_output_path
#      The location in the build tree to copy/symlink python modules Depends on
#      the value of `kwiver_python_subdir`.
#      (e.g. build/lib/python2.7, build/lib/python3.5m)
#
#    kwiver_python_install_path
#      The base location in the install tree where python files/modules are
#      to be installed.
#      (e.g. ${CMAKE_INSTALL_PREFIX}/lib/python3)
#
#    sprokit_python_output_path
#      Similar to `kwiver_python_output_path`. Used by sprokit to define extra
#      python output paths. This may be removed in the future.
#      (e.g. build/lib)
#


###
# Private helper function to execute `python -c "<cmd>"`
#
# Runs a python command and populates an outvar with the result of stdout.
# Be careful of indentation if `cmd` is multiline.
#
# Additional arguments are passed to the underlying `execute_process` function.
#
# Parameters:
#   outvar:
#       The output variable to populate with the content of stdout
#       resulting from the command run.
#   cmd:
#       The command to be run. This will be wrapped within double-quotes. Be
#       careful about indentation if this is to be a multiline command.
#
function( _pycmd outvar cmd )
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "${cmd}"
    RESULT_VARIABLE _exitcode
    OUTPUT_VARIABLE _output
    ${ARGN}
    )
  if( NOT ${_exitcode} EQUAL 0 )
    # Yes, the truple-quotes being escaped and indented the way they are
    # is intentional for the purpose of terminal-based readability.
    message(ERROR "Failed when running python code:
\"\"\"
${cmd}
\"\"\"")
    message( FATAL_ERROR "Python command failed with error code: ${_exitcode}" )
  endif()
  # Remove supurflous newlines (artifacts of print)
  string( STRIP "${_output}" _output )
  set( ${outvar} "${_output}" PARENT_SCOPE )
endfunction()


###
# Private helper function to check if a python package is installed
#
function( _ensure_pypackage_exists package )
  # kwiver_python_install_path may point to site-packages under
  # CMAKE_INSTALL_PREFIX (e.g. when built as part of a larger super-build with
  # packages installed via pip --user / PYTHONUSERBASE).  Add it to
  # sys.path so the import check can find those packages.
  if( kwiver_python_install_path )
    set( _import_cmd "import sys; sys.path.insert(0, '${kwiver_python_install_path}'); import ${package}" )
  else()
    set( _import_cmd "import ${package}" )
  endif()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "${_import_cmd}"
    RESULT_VARIABLE _exitcode
    OUTPUT_VARIABLE _output
    )
  if( NOT ${_exitcode} EQUAL 0 )
    # Indentation for the continued line is intentionally left-justified.
    message( FATAL_ERROR "${package} is missing !
Please install ${package} in the python virtual environment associated with the build" )
  endif()
endfunction()


###
# Python interpreter and libraries
#
# Python 3 is a requirement so we only consider finding at least 3.8 as a
# minimum version.
# PyBind11 is compatible with this find usage.
#
set( _requested_python_components Interpreter Development.Module )
if( (NOT SKBUILD) OR MSVC )
  set( _requested_python_components ${_requested_python_components} Development.Embed )
endif()
find_package( Python 3.8 REQUIRED COMPONENTS ${_requested_python_components} )

# Recording the specific python version that is being utilized.
_pycmd( PYTHON_VERSION "import sys; import re; print(re.match(r'^[0-9]+\.[0-9]+', sys.version)[0])" )
set( KWIVER_PYTHON_VERSION "${PYTHON_VERSION}" CACHE STRING "" )
mark_as_advanced( KWIVER_PYTHON_VERSION )


###
# Python site-packages
#
# Get canonical directory for python site packages (relative to install
# location). It varies from system to system.
#
_pycmd( python_site_packages [==[
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
    rel_path = rel_path[6:]  # Remove "local/"
print(rel_path)
]==] )

# Current usage determines most of the path in alternate ways.
# All we need to supply is the '*-packages' directory name.
# Customers could be converted to accept a larger part of the path from this function.
get_filename_component( python_sitename "${python_site_packages}" NAME )

###
# Python ABI Flags
#
# See PEP 3149 - ABI (application binary interface) version tagged .so files
# https://www.python.org/dev/peps/pep-3149/
#
_pycmd( _python_abi_flags "import sysconfig; print(sysconfig.get_config_var('ABIFLAGS'))" )
set( PYTHON_ABIFLAGS "${_python_abi_flags}"
     CACHE STRING "The ABI flags for the version of Python being used" )
mark_as_advanced( PYTHON_ABIFLAGS )


###
# Find PyBind11 package
#
find_package( pybind11 CONFIG REQUIRED )


###
# Python install path
set( kwiver_python_install_path "${CMAKE_INSTALL_PREFIX}/${python_site_packages}" )



###
# Python package build locations
#
# defines paths used to determine where the kwiver/sprokit/vital python
# packages will be generated in the build tree. (TODO: python modules should
# use a setup.py file to install themselves to the right location)
#
# Instead of constructing the directory with ABIFLAGS just use what python
# gives us.
#
get_filename_component(python_lib_subdir ${python_site_packages} DIRECTORY)  # E.g. "lib/python-3.8"
get_filename_component(python_subdir ${python_lib_subdir} NAME)              # E.g. "python-3.8"
set(kwiver_python_subdir ${python_subdir})
set(kwiver_python_output_path "${KWIVER_BINARY_DIR}/${python_lib_subdir}")

# Currently needs to be separate because sprokit may have CONFIGURATIONS that
# are placed between lib and `kwiver_python_subdir`
set(sprokit_python_output_path "${KWIVER_BINARY_DIR}/lib")


###
# Status string for debugging
#
set( PYTHON_CONFIG_STATUS "
PYTHON_CONFIG_STATUS

  * Python_EXECUTABLE = \"${Python_EXECUTABLE}\"
  * Python_INCLUDE_DIRS = \"${Python_INCLUDE_DIRS}\"
  * Python_LIBRARIES = \"${Python_LIBRARIES}\"

  * PYTHON_ABIFLAGS = \"${PYTHON_ABIFLAGS}\"
  * PYTHON_VERSION = \"${PYTHON_VERSION}\"

  * python_site_packages = \"${python_site_packages}\"
  * python_sitename = \"${python_sitename}\"

  * kwiver_python_subdir = \"${kwiver_python_subdir}\"
  * kwiver_python_install_path = \"${kwiver_python_install_path}\"
  * kwiver_python_output_path = \"${kwiver_python_output_path}\"
  * sprokit_python_output_path = \"${sprokit_python_output_path}\"
")
message( STATUS "${PYTHON_CONFIG_STATUS}" )


###
# Checking for required python packages to support building (and testing)
# python components.
#
_ensure_pypackage_exists("pygccxml")
if (KWIVER_ENABLE_TESTS)
  _ensure_pypackage_exists("pytest")
endif()
