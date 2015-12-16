#
# Precache settings for building fletch
#
# Selects required modules
#
set(fletch_ENABLE_ALL_PACKAGES  OFF CACHE BOOL "" FORCE)
set(fletch_ENABLE_Boost         ON  CACHE BOOL "" FORCE)
set(fletch_ENABLE_Ceres         ON  CACHE BOOL "" FORCE)
set(fletch_ENABLE_Eigen         ON  CACHE BOOL "" FORCE)
set(fletch_ENABLE_OpenCV        ON  CACHE BOOL "" FORCE)
set(fletch_ENABLE_PNG           ON  CACHE BOOL "" FORCE)
set(fletch_ENABLE_SuiteSparse   ON  CACHE BOOL "" FORCE)
set(fletch_ENABLE_TinyXML       ON  CACHE BOOL "" FORCE)
set(fletch_ENABLE_Zlib          ON  CACHE BOOL "" FORCE)
set(fletch_ENABLE_libjpeg-turbo ON  CACHE BOOL "" FORCE)
set(fletch_ENABLE_libtiff       ON  CACHE BOOL "" FORCE)
set(FLETCH_BUILD_WITH_PYTHON    ON  CACHE BOOL "" FORCE)
