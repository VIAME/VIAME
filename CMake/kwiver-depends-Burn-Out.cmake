# Optionally find and configure CUDA dependency

option( KWIVER_ENABLE_BURNOUT
  "Enable Burn-Out dependent code and plugins (Arrows)"
  OFF
  )
mark_as_advanced( KWIVER_ENABLE_BURNOUT )

if( KWIVER_ENABLE_BURNOUT )
  find_package( vidtk REQUIRED )
endif( KWIVER_ENABLE_BURNOUT )
