###
# Find Matlab
#

OPTION( KWIVER_ENABLE_MATLAB
  "Enable Matlab dependent code and plugins (Requires a Matlab installation)"
  OFF
  )

if (KWIVER_ENABLE_MATLAB)
  find_package( Matlab REQUIRED
    COMPONENTS
      ENG_LIBRARY MX_LIBRARY
    )
  # Provides  ${Matlab_ENG_LIBRARY}
  # Provides  ${Matlab_MX_LIBRARY}
  # Provides  ${Matlab_LIBRARIES}
  #include_directories( "${Matlab_INCLUDE_DIRS}" )

  get_filename_component( Matlab_LIBRARY_DIR "${Matlab_ENG_LIBRARY}" DIRECTORY )
  set( SET_MATLAB_LD_LIBRARY_PATH "export LD_LIBRARY_PATH=${Matlab_LIBRARY_DIR}:$LD_LIBRARY_PATH" )

endif()
