
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

add_executable( 0_PhaseSymmetry PhaseSymmetry.cxx )
target_link_libraries( 0_PhaseSymmetry ${ITK_LIBRARIES} )

add_executable( 1_NarrowBandPointSet NarrowBandPointSet.cxx )
target_link_libraries( 1_NarrowBandPointSet ${ITK_LIBRARIES} )

add_executable( 2_PointSetRegistration PointSetRegistration.cxx )
target_link_libraries( 2_PointSetRegistration ${ITK_LIBRARIES} )

add_executable( 3_TransformPointSetAndImage TransformPointSetAndImage.cxx )
target_link_libraries( 3_TransformPointSetAndImage ${ITK_LIBRARIES} )

add_executable( SealRegistration SealRegistration.cxx )
target_link_libraries( SealRegistration ${ITK_LIBRARIES} )
