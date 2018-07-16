# Optional find and confgure VXL dependency

option( KWIVER_ENABLE_FFMPEG
  "Enable FFMPEG dependent code and plugins (Arrows)"
  ${fletch_ENABLED_FFMPEG}
  )

if( KWIVER_ENABLE_FFMPEG )
  find_package( FFMPEG REQUIRED )
endif( KWIVER_ENABLE_FFMPEG )
