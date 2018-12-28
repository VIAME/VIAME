
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
