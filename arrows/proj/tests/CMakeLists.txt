project(arrows_test_proj)

include(kwiver-test-setup)

set(test_libraries    kwiver_algo_proj )

#########################
# Algorithms Proj tests #
#########################
kwiver_discover_tests(proj_geo_map test_libraries test_geo_map.cxx)
