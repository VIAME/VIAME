#
# CMake file for process template
#
##++ replace "template" with something more meaningful for your project
project( template_processes )

#++ uncomment the following line if the project has not already included the file.
# include(kwiver-utils) # provides kwiver_* operations

set( sources
  register_processes.cxx

  template_process.cxx
  template_algo_wrapper.cxx
)

set( private_headers
  template_process.h
  template_algo_wrapper.h
)

kwiver_private_header_group( ${private_headers} )

include_directories( ${sprokit_source_dir}/src
                     ${sprokit_binary_dir}/src   # for generated files
                     ${sprokit_source_dir}/processes # for trait support files
                     ${CMAKE_CURRENT_SOURCE_DIR}
                     ${CMAKE_CURRENT_BINARY_DIR} # needed for generated export header
                     )

##++ Find any required packages. OpenCV is include here only as an example
##++ If your process does not need OpenCV, delete the following code and
##++ replace with what is needed.
find_package( OpenCV REQUIRED )
if( OpenCV_VERSION VERSION_LESS "2.4" )
  message( FATAL_ERROR "OpenCV version must be at least 2.4" )
endif()

##++ This builds a plugin containing the specified process(s)
##++ Replace "template_process" with the desired name of the plugin file
kwiver_add_plugin( template_processes
  SUBDIR           ${kwiver_plugin_process_subdir}
  SOURCES          ${sources}
                   ${private_headers}
  PRIVATE          sprokit_pipeline
                   vital vital_vpm vital_logger vital_config
                   kwiver_algo_ocv
                   kwiversys
 )

##++ The above kwiver_add_plugin operation generates a file "template_processes_export.h"
##++ that defines the export related symbols needed in the source code.
