project(vistk_test_python_utilities)

##############################
# Homography tests
##############################
vistk_build_python_test(homography test-homography.py)

vistk_make_python_test(homography import)
vistk_make_python_test(homography api_calls)

##############################
# Timestamp tests
##############################
vistk_build_python_test(timestamp test-timestamp.py)

vistk_make_python_test(timestamp import)
vistk_make_python_test(timestamp api_calls)
