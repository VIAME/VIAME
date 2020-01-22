#
# CMake file for callable sprokit pipeline
#

include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

kwiver_add_library( SMQTK_Descriptor
  SMQTK_Descriptor.h       SMQTK_Descriptor.cxx
  )

target_link_libraries( SMQTK_Descriptor
  PRIVATE         sprokit_pipeline sprokit_pipeline_util
                  vital vital_vpm vital_logger vital_config
                  kwiver_adapter
                  kwiver_algo_ocv
  )


add_executable( SMQTK_Descriptor_test
  SMQTK_Descriptor_test.cxx

  SMQTK_Descriptor.h SMQTK_Descriptor.cxx
  )

target_link_libraries( SMQTK_Descriptor_test
  PRIVATE         sprokit_pipeline sprokit_pipeline_util
                  vital vital_vpm vital_logger vital_config
                  kwiver_adapter
                  kwiver_algo_ocv
  )

install( TARGETS SMQTK_Descriptor_test
  RUNTIME DESTINATION bin
  )
