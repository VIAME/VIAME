project(arrows_test_ffmpeg)

include(kwiver-test-setup)

set(test_libraries     kwiver_algo_ffmpeg)

##############################
# Algorithms ffmpeg tests
##############################
kwiver_discover_gtests(ffmpeg open_video LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}")
