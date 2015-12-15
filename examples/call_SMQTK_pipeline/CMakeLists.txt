#
# CMake file for callable sprokit pipeline
#

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )


kwiver_add_plugin( SMQTK_extract
  SUBDIR           sprokit
  SOURCES            io_mgr.h                 io_mgr.cxx
                     supply_image.h           supply_image.cxx
                     accept_descriptor.h      accept_descriptor.cxx
                     register_processes.cxx

  PRIVATE          sprokit_pipeline sprokit_tools
                   maptk_core maptk_vxl maptk_ocv
                   vital vital_apm vital_logger vital_config
                   ${OpenCV_LIBS}
                   ${Boost_SYSTEM_LIBRARY}
  )

add_library( SMQTK_Descriptor
  SMQTK_Descriptor.h       SMQTK_Descriptor.cxx
  io_mgr.h                 io_mgr.cxx
  )

target_link_libraries( SMQTK_Descriptor
  PRIVATE         sprokit_pipeline sprokit_tools
#                  maptk_core maptk_vxl maptk_ocv
                  vital vital_apm vital_logger vital_config
                  ${OpenCV_LIBS}
                  ${Boost_SYSTEM_LIBRARY}
  )


add_executable( desc_test
  SMQTK_Descriptor_test.cxx

  io_mgr.h           io_mgr.cxx
  SMQTK_Descriptor.h SMQTK_Descriptor.cxx
  )

target_link_libraries( desc_test
  PRIVATE             # SMQTK_extract
                      sprokit_pipeline sprokit_tools
                      ${OpenCV_LIBS}
                      ${Boost_SYSTEM_LIBRARY}
  )

install( TARGETS desc_test
  RUNTIME DESTINATION bin
  )
