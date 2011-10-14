project(vistk_processes_helpers_video)

set(video_helper_srcs
  istream.cxx)

add_library(video_helper STATIC
  ${video_helper_srcs})
target_link_libraries(video_helper
  image_helper
  vidl)

if (NOT WIN32)
  set_target_properties(video_helper
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif (NOT WIN32)
