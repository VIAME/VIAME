project(vistk_test_python_pipeline_util)

##############################
# Bake tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-bake.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-bake
  @ONLY)

vistk_declare_test(python-bake)

vistk_make_python_test(python-bake import)

##############################
# Export tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-export_.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-export_
  @ONLY)

vistk_declare_test(python-export_)

vistk_make_python_test(python-export_ import)

##############################
# Load tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-load.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-load
  @ONLY)

vistk_declare_test(python-load)

vistk_make_python_test(python-load import)
