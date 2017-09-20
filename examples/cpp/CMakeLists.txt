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

###
# include(${kwiver_DIR}/UseKWIVER.cmake)
#
find_package( fletch NO_MODULE )
if ( fletch_FOUND )
  # Search for libraries in the fletch install root
  list(APPEND CMAKE_PREFIX_PATH "${fletch_ROOT}")
endif()
include_directories(${fletch_ROOT}/include)

if(VXL_DIR)
  find_package(VXL)
  if(VXL_FOUND)
    include(${VXL_CMAKE_DIR}/UseVXL.cmake)
  else()
    message(FATAL_ERROR "VXL_DIR should be built by fletch.")
  endif()
endif()

if(OpenCV_DIR)
  find_package(OpenCV)
  if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIR})
    set(KWIVER_LIBRARIES ${KWIVER_LIBRARIES} ${OpenCV_LIBS})
  else()
    message(FATAL_ERROR "OpenCV_DIR should be built by fletch.")
  endif()
endif()

include_directories(${KWIVER_INCLUDE_DIRS})
link_directories(${KWIVER_LIBRARY_DIRS})
#
# end UseKWIVER.cmake
###

set(SOURCE "main.cpp"
           "how_to_use_images.cpp"
)

add_executable(HowToDriver ${SOURCE})
target_link_libraries(HowToDriver ${KWIVER_LIBRARIES})

