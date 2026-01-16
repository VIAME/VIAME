
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
  ITKTransform.h
  RegisterOpticalAndThermal.h
)

set( plugin_sources
  ${plugin_sources}
  ITKTransform.cxx
  RegisterOpticalAndThermal.cxx
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

target_include_directories( viame_itk PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_itk
  PUBLIC               kwiver::vital kwiver::vital_algo kwiver::vital_config
                       kwiver::vital_exceptions kwiver::vital_logger kwiver::vital_util
                       kwiver::kwiversys
                       ITKCommon
  PRIVATE              ITKLabelMap ITKPath ITKOptimizersv4 ITKOptimizers
                       ITKStatistics ITKTransform ITKSpatialObjects
                       ITKEXPAT ITKMesh ITKQuadEdgeMesh ITKIOMeshFreeSurfer
                       ITKIOMeshVTK ITKIOTransformHDF5 ITKIOTransformBase
                       ITKTransformFactory ITKIOTransformInsightLegacy

                       ITKIOGE ITKIOIPL ITKIOGIPL ITKIOGDCM
                       ITKIOHDF5 ITKIOJPEG ITKIOJPEG2000 ITKIOLSM
                       ITKIOTIFF ITKIOMINC ITKIOMRC ITKIOMeta
                       ITKIONIFTI ITKIONRRD ITKIOPNG ITKIOStimulate ITKIOVTK
                       ITKIOMeshOFF ITKIOMeshOBJ ITKIOMeshBYU ITKIOMeshBase
                       ITKIOMeshGifti ITKIOTransformMatlab ITKIOBMP ITKIOBioRad
                       ITKIOBruker ITKIOImageBase ITKIOSpatialObjects
  )

set_target_properties( viame_itk PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_itk
  RegisterPlugins.cxx
  )

target_link_libraries( viame_itk_plugin
  PUBLIC               kwiver::vital_vpm
  )


# Add KWIVER plugin registration
if( VIAME_ENABLE_VXL AND VIAME_ENABLE_OPENCV )

  set( process_sources
    ../core/align_multimodal_imagery_process.cxx
    RegisterProcesses.cxx
    RegistrationProcess.cxx
    WarpDetectionsProcess.cxx
    WarpImageProcess.cxx
  )

  set( process_headers
    ../core/align_multimodal_imagery_process.h
    RegistrationProcess.h
    WarpDetectionsProcess.h
    WarpImageProcess.h
  )

  kwiver_add_plugin( viame_processes_itk
    SUBDIR           ${viame_plugin_process_subdir}
    SOURCES          ${process_sources}
                     ${private_headers}
    PRIVATE          kwiver::sprokit_pipeline
                     viame_itk
                     kwiver::vital kwiver::vital_vpm kwiver::vital_logger kwiver::vital_config
                     kwiver::kwiver_algo_core kwiver::kwiver_algo_ocv kwiver::kwiversys
                     ${OpenCV_LIBS}
                     ITKIOTransformBase ITKTransformFactory
                     ITKIOTransformInsightLegacy
                     ITKIOHDF5 ITKIOTransformHDF5 ITKIOTransformMatlab
   )

  target_include_directories( viame_processes_itk PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../core>
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

#kwiver_add_executable( itk_eo_ir_registration RegistrationTool.cxx )
#target_link_libraries( itk_eo_ir_registration viame_itk )

kwiver_add_executable( itk_point_set_to_transform PointSetToTransform.cxx )
target_link_libraries( itk_point_set_to_transform ${ITK_LIBRARIES} )
