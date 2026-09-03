
string( REPLACE "." ";" Python_VERSION_LIST ${VIAME_PYTHON_VERSION} )
list( GET Python_VERSION_LIST 0 Python_VERSION_MAJOR )
list( GET Python_VERSION_LIST 1 Python_VERSION_MINOR )
list( GET Python_VERSION_LIST 2 Python_VERSION_PATCH )

set( Python_VERSION ${VIAME_PYTHON_VERSION} CACHE INTERNAL "Forced" FORCE )
set( Python_FOUND TRUE CACHE INTERNAL "Forced" FORCE )
set( Python_SOABI ""   CACHE INTERNAL "Forced" FORCE )

set( Python_INCLUDE_DIR ${VIAME_INSTALL_PREFIX}/include CACHE PATH "Forced" FORCE )

if( WIN32 )
  set( Python_EXECUTABLE ${VIAME_INSTALL_PREFIX}/bin/python.exe CACHE PATH "Forced" FORCE )

  # Use the version specific import library rather than the stable ABI
  # python3.lib. CMake's FindPython locates the matching runtime DLL from the
  # library name, and it cannot resolve one for python3.lib, which makes the
  # Development.Module and Development.Embed components report as missing in
  # downstream projects such as kwiver.
  set( Python_LIBRARY
    ${VIAME_INSTALL_PREFIX}/lib/python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.lib
    CACHE PATH "Forced" FORCE )
else()
  set( Python_EXECUTABLE ${VIAME_INSTALL_PREFIX}/bin/python CACHE PATH "Forced" FORCE )
  set( Python_LIBRARY ${VIAME_INSTALL_PREFIX}/lib/libpython3.so CACHE PATH "Forced" FORCE )
endif()
