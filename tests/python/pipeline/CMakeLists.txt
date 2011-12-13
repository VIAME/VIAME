project(vistk_test_python_pipeline)

##############################
# Config tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-config.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-config
  @ONLY)

vistk_declare_test(python-config)

vistk_make_python_test(python-config import)
vistk_make_python_test(python-config create)
