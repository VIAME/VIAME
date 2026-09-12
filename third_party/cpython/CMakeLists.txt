# CPython, built from source when VIAME has to ship its own interpreter
#
# Optional and off by default. `VIAME_BUILD_PYTHON_FROM_SOURCE` turns it on,
# and what wants it is a packaged build -- an installer that cannot assume a
# python on the target machine. Every other build uses the system one, which
# is what `find_package( Python )` finds.
#
# Not vendored, fetched: CPython's source is 25 MB of tarball and belongs in
# the repository no more than googletest's does. What is here is the recipe.
#
# `cmake/setup_internal_python.cmake` already declares where the interpreter
# will be -- `<prefix>/bin/python`, `<prefix>/lib/libpythonX.Y.so`,
# `<prefix>/include/pythonX.Y` -- and asserts rather than probes, so this
# only has to put it there. That file needed no changes.
#
# What this does **not** carry over from fletch's recipe, deliberately:
#
#   the second, static build   fletch builds CPython twice on Linux, shared
#                              and then static, so the installed `python` is
#                              a static executable. That is a startup-time
#                              optimisation costing a full second compile of
#                              CPython; the shared interpreter is what VIAME
#                              links against either way.
#   the pip bootstrap and      `cmake/add_project_python_deps.cmake` is where
#   the package installs       VIAME installs its own python dependencies,
#                              and it does not care who built the
#                              interpreter. `ensurepip` runs here and stops.

if( NOT VIAME_BUILD_PYTHON_FROM_SOURCE )
  return()
endif()

include( ExternalProject )

# `3.14.2t` and its kind: the `t` suffix is a free-threaded build
set( _cpython_version "${VIAME_PYTHON_VERSION}" )
set( _cpython_free_threaded OFF )

if( _cpython_version MATCHES "t$" )
  set( _cpython_free_threaded ON )
  string( REGEX REPLACE "t$" "" _cpython_version "${_cpython_version}" )
endif()

set( _cpython_url
  "https://www.python.org/ftp/python/${_cpython_version}/Python-${_cpython_version}.tgz" )

if( WIN32 )
  # Windows builds through `PCBuild/build.bat` rather than autotools, and the
  # install is a file copy driven by a script. This is the path an MSI needs
  # and the one that cannot be exercised here; until it has been, a Windows
  # build with `VIAME_BUILD_PYTHON_FROM_SOURCE` on should keep using fletch's
  # CPython, which is why this refuses rather than half-working.
  message( FATAL_ERROR
    "third_party/cpython builds the autotools path only. On Windows, leave "
    "VIAME_BUILD_PYTHON_FROM_SOURCE to fletch's CPython until this recipe "
    "has been exercised there." )
endif()

set( _cpython_configure_args
  --prefix=${VIAME_INSTALL_PREFIX}
  --enable-shared
  --enable-optimizations
  --with-ensurepip=install
  # The interpreter has to be able to find its own `libpython`.
  # `--enable-shared` alone does not arrange that for a prefix outside the
  # system paths: the built `python` links `libpython3.X.so.1.0`, records no
  # RPATH, and dies at startup with "cannot open shared object file" unless
  # something has already put the prefix on `LD_LIBRARY_PATH`. Found by
  # running it. This is the problem fletch's recipe avoids by building
  # CPython a **second** time statically -- a whole extra compile of CPython
  # to solve what a linker flag solves.
  "LDFLAGS=-Wl,-rpath,${VIAME_INSTALL_PREFIX}/lib"
  )

if( _cpython_version VERSION_GREATER_EQUAL "3.8" )
  list( APPEND _cpython_configure_args --with-openssl=/usr )
endif()

if( _cpython_free_threaded )
  list( APPEND _cpython_configure_args --disable-gil )
endif()

if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
  list( APPEND _cpython_configure_args --with-pydebug )
endif()

find_program( MAKE_EXECUTABLE NAMES gmake make REQUIRED )

ExternalProject_Add( viame_cpython
  URL                 ${_cpython_url}
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  PREFIX              ${CMAKE_CURRENT_BINARY_DIR}
  BUILD_IN_SOURCE     1
  CONFIGURE_COMMAND   ./configure ${_cpython_configure_args}
  BUILD_COMMAND       ${MAKE_EXECUTABLE} -j${VIAME_PYTHON_BUILD_JOBS}
  # `altinstall` would leave no `python`, and the rest of the tree asks for
  # one by that name. `install` writes both `python3` and `python3.X`.
  INSTALL_COMMAND     ${MAKE_EXECUTABLE} install
  USES_TERMINAL_BUILD 1
  )

# `python` and `pip` without the version suffix, which `setup_viame.sh` and
# every script in `tools/` expect to exist
ExternalProject_Add_Step( viame_cpython unsuffixed_names
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          python3 ${VIAME_INSTALL_PREFIX}/bin/python
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          pip3 ${VIAME_INSTALL_PREFIX}/bin/pip
  DEPENDEES install
  )

# Anything compiling against `Python.h` has to wait for this. The interpreter
# is a build input here rather than a prior superbuild stage, which is the
# one thing moving it into the tree changes.
add_library( viame_python_interpreter INTERFACE )
add_dependencies( viame_python_interpreter viame_cpython )
add_library( viame::python_interpreter ALIAS viame_python_interpreter )
