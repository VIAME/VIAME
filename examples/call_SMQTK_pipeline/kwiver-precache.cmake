#
# Precache settings for building kwiver to support the SMQTK/C++ Bridge
#
# Selects required modules
#
set(CMAKE_BUILD_TYPE                Release CACHE STRING "" FORCE)
set(KWIVER_BUILD_KWIVER_PROCESSES   ON      CACHE BOOL "" FORCE)
set(KWIVER_BUILD_SHARED             ON      CACHE BOOL "" FORCE)
set(KWIVER_ENABLE_MAPTK             ON      CACHE BOOL "" FORCE)
set(KWIVER_ENABLE_OPENCV            ON      CACHE BOOL "" FORCE)
set(KWIVER_ENABLE_PYTHON            ON      CACHE BOOL "" FORCE)
