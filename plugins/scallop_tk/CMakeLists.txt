###
# Loadable algorithms

find_package( ScallopTK REQUIRED )

## These should come from the find package.
set( ScallopTK_LIBS ScallopTK )

include_directories( "${ScallopTK_INCLUDE_DIR}" )
link_directories( "${ScallopTK_LIBRARY_DIR}" )

set( plugin_headers
  scallop_tk_detector.h
  )

set( plugin_sources
  scallop_tk_detector.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_scallop_tk_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_scallop_tk
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( viame_scallop_tk
  PUBLIC               vital
                       kwiver_algo_ocv
                       ScallopTK
  )
  
if(WIN32)
  target_link_libraries( viame_scallop_tk
  PUBLIC			     caffe
					     GLog
					     GFlags
					     proto
					     libprotobuf
					     caffehdf5
					     caffehdf5_hl
					     boost_system-vc140-mt-1_55
					     boost_thread-vc140-mt-1_55
					     boost_filesystem-vc140-mt-1_55
					     libopenblas
					     leveldb
					     lmdb
					     lmdbd
					     ntdll
					   )
  if(VIAME_ENABLE_CUDA)
    target_link_libraries( viame_scallop_tk PUBLIC cudart)
  endif()
endif()

algorithms_create_plugin( viame_scallop_tk
  register_algorithms.cxx
  )
