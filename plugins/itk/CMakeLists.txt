
find_package( ITK 5.0 REQUIRED
  COMPONENTS
    AnisotropicDiffusionLBR
    PhaseSymmetry
    ITKCommon
    ITKImageIO
    ITKMeshIO
    ITKTransformIO
    ITKMetricsv4
    ITKLevelSets)
include( ${ITK_USE_FILE} )

kwiver_add_executable( itk_0_phase_symmetry PhaseSymmetry.cxx )
target_link_libraries( itk_0_phase_symmetry ${ITK_LIBRARIES} )

kwiver_add_executable( itk_1_narrow_band_point_set NarrowBandPointSet.cxx )
target_link_libraries( itk_1_narrow_band_point_set ${ITK_LIBRARIES} )

kwiver_add_executable( itk_2_point_set_registration PointSetRegistration.cxx )
target_link_libraries( itk_2_point_set_registration ${ITK_LIBRARIES} )

kwiver_add_executable( itk_3_transform_point_set_and_image TransformPointSetAndImage.cxx )
target_link_libraries( itk_3_transform_point_set_and_image ${ITK_LIBRARIES} )

kwiver_add_executable( itk_seal_registration SealRegistration.cxx )
target_link_libraries( itk_seal_registration ${ITK_LIBRARIES} )

if( VIAME_ENABLE_VXL )

  set( plugin_headers
    SealRegistrationProcess.h
    )

  set( plugin_sources
    SealRegistrationProcess.cxx
    )

  kwiver_install_headers(
    SUBDIR     viame
    ${plugin_headers}
    )

  kwiver_install_headers(
    ${CMAKE_CURRENT_BINARY_DIR}/viame_itk_export.h
    NOPATH   SUBDIR     viame
    )

  kwiver_add_library( viame_itk
    ${plugin_headers}
    ${plugin_sources}
    )

  target_link_libraries( viame_itk
    PUBLIC               vital
                         vital_algo
                         vital_config
                         vital_exceptions
                         vital_logger
                         vital_util
                         kwiversys
    )

  set_target_properties( viame_itk PROPERTIES
    SOVERSION            ${VIAME_VERSION_MAJOR}
    )

  algorithms_create_plugin( viame_itk
    RegisterAlgorithms.cxx
    )

  target_link_libraries( viame_itk_plugin
    PUBLIC               vital_vpm
    )

endif()
