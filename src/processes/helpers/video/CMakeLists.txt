project(vistk_processes_helpers_video)

set(video_helper_srcs
  istream.cxx)

vistk_add_helper_library_sources(video_helper video_helper_srcs
  image_helper
  vidl)
