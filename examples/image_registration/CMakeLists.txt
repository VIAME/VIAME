###
# Image Enhancement and Filtering Example
##
set( EXAMPLE_INSTALL_FILES
     README.rst )

if( VIAME_ENABLE_ITK OR VIAME_ENABLE_OPENCV )
  set( EXAMPLE_INSTALL_FILES
    ${EXAMPLE_INSTALL_FILES}
    input_optical_images.txt
    input_thermal_images.txt )
endif()

if( VIAME_ENABLE_OPENCV )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
      ${EXAMPLE_INSTALL_FILES}
      generate_mosaic_for_folder.bat
      generate_mosaic_for_list.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
      ${EXAMPLE_INSTALL_FILES}
      generate_mosaic_for_folder.sh
      generate_mosaic_for_list.sh )
  endif()
endif()

if( VIAME_ENABLE_ITK )
  if( WIN32 )
    set( EXAMPLE_INSTALL_FILES
      ${EXAMPLE_INSTALL_FILES}
      register_eo_ir_per_frame_itk.bat
      launch_point_alignment_interface.bat
      generate_transform_from_points.bat )
  else()
    set( EXAMPLE_INSTALL_FILES
      ${EXAMPLE_INSTALL_FILES}
      register_eo_ir_per_frame_itk.sh
      launch_point_alignment_interface.sh
      generate_transform_from_points.sh )
  endif()

  if( VIAME_ENABLE_OPENCV )
    if( WIN32 )
      set( EXAMPLE_INSTALL_FILES
        ${EXAMPLE_INSTALL_FILES}
        register_eo_ir_per_frame_ocv.bat )
    else()
      set( EXAMPLE_INSTALL_FILES
        ${EXAMPLE_INSTALL_FILES}
        register_eo_ir_per_frame_ocv.sh )
    endif()
  endif()
endif()

###
# Install pipelines to example directory
##
install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/image_registration )
