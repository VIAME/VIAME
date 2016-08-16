###
# Find Matlab
#

OPTION( KWIVER_ENABLE_MATLAB
  "Enable matlab dependent code and plugins"
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
endif()
