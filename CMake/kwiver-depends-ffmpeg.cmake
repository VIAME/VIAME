# Optional find and confgure VXL dependency

option( KWIVER_ENABLE_FFMPEG
  "Enable FFmpeg dependent code and plugins (Arrows)"
  ${fletch_ENABLED_FFmpeg}
  )

if( KWIVER_ENABLE_FFMPEG )
  find_package( FFMPEG 3.0  REQUIRED )
endif( KWIVER_ENABLE_FFMPEG )
