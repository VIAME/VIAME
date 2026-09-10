# Central location for KWIVER external dependency declaration and resolution

# On macOS, prefer unix-style packages (e.g. from Fletch) over Frameworks
# when looking for dependencies
set(CMAKE_FIND_FRAMEWORK LAST)

# Required for Vital
include( kwiver-depends-Eigen )

# Optional for Vital (loggers)
include( kwiver-depends-log4cxx )
include( kwiver-depends-log4cplus )

# Required for Sprokit
if(KWIVER_ENABLE_SPROKIT
    OR (KWIVER_ENABLE_TOOLS AND NOT VITAL_USE_STD_REGEX))
  include( kwiver-depends-Boost )
endif()

# Optional for Arrows
if(KWIVER_ENABLE_ARROWS)
  include( kwiver-depends-CUDA )
  include( kwiver-depends-OpenCV )
  include( kwiver-depends-DBoW2 )
  include( kwiver-depends-PROJ )
  include( kwiver-depends-Ceres )
  include( kwiver-depends-Qt )
  include( kwiver-depends-VXL )
  include( kwiver-depends-VTK )
  include( kwiver-depends-uuid )
  include( kwiver-depends-kpf )
  include( kwiver-depends-ffmpeg )
  include( kwiver-depends-GDAL )
  include( kwiver-depends-PDAL )
  include( kwiver-depends-PyTorch )
  include( kwiver-depends-zlib )
endif()

include( kwiver-depends-Matlab )
include( kwiver-depends-ZeroMQ )
include( kwiver-depends-OpenMP )
