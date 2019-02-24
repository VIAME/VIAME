
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

# Add viame ITK library
set( plugin_headers
  ${plugin_headers}
  RegisterOpticalAndThermal.h
  )

set( plugin_sources
  ${plugin_sources}
  RegisterOpticalAndThermal.cxx
  )

if( VIAME_ENABLE_VXL AND VIAME_ENABLE_OPENCV )
  set( plugin_headers
    ${plugin_headers}
    RegistrationProcess.h
    )

  set( plugin_sources
    ${plugin_sources}
    RegistrationProcess.cxx
    )
endif()

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
  PUBLIC               vital vital_algo vital_config
                       vital_exceptions vital_logger vital_util
                       kwiversys ITKCommon
  PRIVATE              ITKLabelMap ITKPath ITKOptimizersv4 ITKOptimizers
                       ITKStatistics ITKTransform ITKSpatialObjects
                       ITKEXPAT ITKMesh ITKQuadEdgeMesh ITKIOMeshFreeSurfer
                       ITKIOMeshVTK ITKIOTransformHDF5 ITKIOTransformBase
                       ITKTransformFactory ITKIOTransformInsightLegacy

                       ITKIOGE ITKIOIPL ITKIOGIPL ITKIOGDCM
                       ITKIOHDF5 ITKIOJPEG ITKIOJPEG2000 ITKIOLSM
                       ITKIOTIFF ITKIOMINC ITKIOMRC ITKIOMeta
                       ITKIONIFTI ITKIONRRD ITKIOPNG ITKIOStimulate
                       ITKIOVTK ITKIOMesh ITKIOMeshBYU ITKIOMeshBase
                       ITKIOTransformMatlab ITKIOBMP ITKIOBioRad
                       ITKIOBruker ITKIOImageBase ITKIOSpatialObjects
  )

set_target_properties( viame_itk PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

# Add KWIVER plugin registration
if( VIAME_ENABLE_VXL AND VIAME_ENABLE_OPENCV )
  algorithms_create_plugin( viame_itk
    RegisterAlgorithms.cxx
    )

  target_link_libraries( viame_itk_plugin
    PUBLIC               vital_vpm
    )
endif()

# Add auxiliary standalone tools
kwiver_add_executable( itk_0_phase_symmetry PhaseSymmetry.cxx )
target_link_libraries( itk_0_phase_symmetry ${ITK_LIBRARIES} )

kwiver_add_executable( itk_1_narrow_band_point_set NarrowBandPointSet.cxx )
target_link_libraries( itk_1_narrow_band_point_set ${ITK_LIBRARIES} )

kwiver_add_executable( itk_2_point_set_registration PointSetRegistration.cxx )
target_link_libraries( itk_2_point_set_registration ${ITK_LIBRARIES} )

kwiver_add_executable( itk_3_transform_point_set_and_image TransformPointSetAndImage.cxx )
target_link_libraries( itk_3_transform_point_set_and_image ${ITK_LIBRARIES} )

kwiver_add_executable( itk_eo_ir_registration RegistrationTool.cxx )
target_link_libraries( itk_eo_ir_registration viame_itk )
