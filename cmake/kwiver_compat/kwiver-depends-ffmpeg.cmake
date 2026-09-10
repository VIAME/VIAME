# Optional find and configure FFmpeg dependency

kwiver_package_option(FFMPEG
  DESCRIPTION "Enable FFmpeg dependent code and plugins (Arrows)"
  FLETCH_NAME FFmpeg
)

if( kwiver_enabled_ffmpeg )
  find_package( FFMPEG 3.0  REQUIRED )
endif()
