project(vistk_processes_helpers_video)

set(video_helper_srcs
  istream.cxx)

set(video_helper_headers
  istream.h)

set(video_helper_all_srcs
  ${video_helper_srcs}
  ${video_helper_headers})

vistk_add_helper_library_sources(video_helper video_helper_all_srcs
  image_helper
  vidl)
