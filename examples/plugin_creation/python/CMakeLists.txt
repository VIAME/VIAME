cmake_minimum_required( VERSION 3.15 )
project( ExampleExternalPythonPlugin NONE )

# Where the VIAME install is
set( VIAME_DIR "" CACHE PATH "Location of VIAME installation" )

if( NOT VIAME_DIR OR NOT EXISTS "${VIAME_DIR}/setup_viame.sh" AND NOT EXISTS "${VIAME_DIR}/setup_viame.bat" )
  message( FATAL_ERROR "Set VIAME_DIR to a VIAME installation" )
endif()

# The install's own site-packages, whatever python it was built with
file( GLOB _site_packages LIST_DIRECTORIES TRUE "${VIAME_DIR}/lib/python3*/site-packages" )
if( NOT _site_packages )
  message( FATAL_ERROR "No lib/python3*/site-packages under ${VIAME_DIR}" )
endif()
list( GET _site_packages 0 _site_packages )

# The package name is what `VIAME_PYTHON_PLUGINS` names
install( FILES __init__.py example_filter.py example_filter_process.py
         DESTINATION "${_site_packages}/example_external_plugin" )
