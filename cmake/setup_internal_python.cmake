
set( VIAME_PYTHON_VERSION 3.10.4 CACHE STRING "Select the version of Python to build." )
set_property( CACHE VIAME_PYTHON_VERSION PROPERTY STRINGS "3.6.15" "3.8.13" "3.10.4" )
mark_as_advanced( VIAME_PYTHON_VERSION )

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
  set( Python_LIBRARY ${VIAME_INSTALL_PREFIX}/lib/python3.lib CACHE PATH "Forced" FORCE )
else()
  set( Python_EXECUTABLE ${VIAME_INSTALL_PREFIX}/bin/python CACHE PATH "Forced" FORCE )
  set( Python_LIBRARY ${VIAME_INSTALL_PREFIX}/lib/libpython3.so CACHE PATH "Forced" FORCE )
endif()
