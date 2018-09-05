project(arrows_test_gdal)

include(kwiver-test-setup)

set(test_libraries      vital vital_vpm kwiver_algo_gdal )

##############################
# Algorithms GDAL tests
##############################
kwiver_discover_gtests(gdal image  LIBRARIES ${test_libraries} ARGUMENTS "${kwiver_test_data_directory}")
