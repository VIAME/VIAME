# Central location for KWIVER external dependency declaration and resolution

# Required for Vital
include( kwiver-depends-Eigen )

# Optional for Vital (loggers)
include( kwiver-depends-log4cxx )
include( kwiver-depends-log4cplus )

# Required for Sprokit and Track Oracle
if(KWIVER_ENABLE_SPROKIT OR KWIVER_ENABLE_TRACK_ORACLE)
  include( kwiver-depends-Boost )
endif()

# Required for Track Oracle
if(KWIVER_ENABLE_TRACK_ORACLE)
  include( kwiver-depends-TinyXML )
  include( kwiver-depends-VXL )
endif()

# Optional for Arrows
if(KWIVER_ENABLE_ARROWS)
  include( kwiver-depends-CUDA )
  include( kwiver-depends-OpenCV )
  include( kwiver-depends-PROJ )
  include( kwiver-depends-VisCL )
  include( kwiver-depends-Ceres )
  include( kwiver-depends-VXL )
  include( kwiver-depends-Matlab )
  include( kwiver-depends-darknet )
  include( kwiver-depends-database )
  include( kwiver-depends-Burn-Out )
  include( kwiver-depends-uuid )
  include( kwiver-depends-kpf )
endif()
