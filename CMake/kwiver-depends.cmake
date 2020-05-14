# Central location for KWIVER external dependency declaration and resolution

# On macOS, prefer unix-style packages (e.g. from Fletch) over Frameworks
# when looking for dependencies
set(CMAKE_FIND_FRAMEWORK LAST)

# Required for Vital
include( kwiver-depends-Eigen )

# Optional for Vital (loggers)
include( kwiver-depends-log4cxx )
include( kwiver-depends-log4cplus )

# Required for Sprokit and Track Oracle
if(KWIVER_ENABLE_SPROKIT OR KWIVER_ENABLE_TRACK_ORACLE
    OR (KWIVER_ENABLE_TOOLS AND NOT VITAL_USE_STD_REGEX))
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
  include( kwiver-depends-Qt )
  include( kwiver-depends-VXL )
  include( kwiver-depends-Matlab )
  include( kwiver-depends-darknet )
  include( kwiver-depends-database )
  include( kwiver-depends-Burn-Out )
  include( kwiver-depends-uuid )
  include( kwiver-depends-kpf )
  include( kwiver-depends-SVM )
  include( kwiver-depends-ffmpeg )
  include( kwiver-depends-GDAL )
  include( kwiver-depends-PyTorch )
endif()

include( kwiver-depends-ZeroMQ )
include( kwiver-depends-OpenMP )
