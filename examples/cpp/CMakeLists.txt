cmake_minimum_required(VERSION 3.3)

project(HowTo-KWIVER)

set( kwiver_DIR "" CACHE PATH "Path to KWIVER packages" )
if ( IS_DIRECTORY ${kwiver_DIR} )
  find_package( kwiver NO_MODULE )
  if ( kwiver_FOUND )
    # Search for libraries in the fletch install root
    list(APPEND CMAKE_PREFIX_PATH "${kwiver_ROOT}")
  else()
    message(FATAL_ERROR "Specify the location of kwiver")
  endif()
else()
  message(FATAL_ERROR "Specify the location of kwiver")
endif()


# TODO Should this code go in kwiver-config.cmake?
# We can always use any package provided by fletch

# We can find fletch and then find any package it provides
find_package( fletch NO_MODULE )
if ( fletch_FOUND )
  # Search for libraries in the fletch install root
  list(APPEND CMAKE_PREFIX_PATH "${fletch_ROOT}")
endif()
include_directories(${fletch_ROOT}/include)

# For this project, we will be using some direct calls to OpenCV
# Let's find it and add it to our project scope
find_package(OpenCV)
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_directories(${OpenCV_LIB_DIR})
  set(KWIVER_LIBRARIES ${KWIVER_LIBRARIES} ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "OpenCV_DIR should be built by fletch.")
endif()


# This should be the only CMake calls you need to include KWIVER in your applicationc
# (Once you find kwiver, as shown above)
include_directories(${KWIVER_INCLUDE_DIRS})
link_directories(${KWIVER_LIBRARY_DIRS})
set(SOURCE "main.cpp"
           "how_to_part_01_images.cpp"
           "how_to_part_02_detections.cpp"
)

add_executable(HowToDriver ${SOURCE})
target_link_libraries(HowToDriver ${KWIVER_LIBRARIES})

