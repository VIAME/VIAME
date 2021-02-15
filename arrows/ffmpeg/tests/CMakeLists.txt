project(arrows_test_ffmpeg)

set(CMAKE_FOLDER "Arrows/FFmpeg/Tests")

include(kwiver-test-setup)

set(test_libraries     kwiver_algo_ffmpeg kwiver_algo_core)

##############################
# Algorithms ffmpeg tests
##############################
kwiver_discover_gtests(ffmpeg video_input_ffmpeg LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}")
