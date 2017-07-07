# Optionally find and configure CUDA dependency

option( KWIVER_ENABLE_CUDA
  "Enable CUDA dependent code and plugins"
  OFF
  )
# This option is currently a hack and does not really control all CUDA use in KWIVER
mark_as_advanced( KWIVER_ENABLE_CUDA )

if( KWIVER_ENABLE_CUDA )
  find_package( CUDA REQUIRED )
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif( KWIVER_ENABLE_CUDA )
