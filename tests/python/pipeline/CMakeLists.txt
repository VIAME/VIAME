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

##############################
# Datum tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-datum.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-datum
  @ONLY)

vistk_declare_test(python-datum)

vistk_make_python_test(python-datum import)

##############################
# Edge tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-edge.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-edge
  @ONLY)

vistk_declare_test(python-edge)

vistk_make_python_test(python-edge import)

##############################
# Modules tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-modules.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-modules
  @ONLY)

vistk_declare_test(python-modules)

vistk_make_python_test(python-modules import)

##############################
# Pipeline tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-pipeline.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-pipeline
  @ONLY)

vistk_declare_test(python-pipeline)

vistk_make_python_test(python-pipeline import)

##############################
# Process tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-process.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-process
  @ONLY)

vistk_declare_test(python-process)

vistk_make_python_test(python-process import)

##############################
# Process registry tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-process_registry.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-process_registry
  @ONLY)

vistk_declare_test(python-process_registry)

vistk_make_python_test(python-process_registry import)

##############################
# Schedule tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-schedule.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-schedule
  @ONLY)

vistk_declare_test(python-schedule)

vistk_make_python_test(python-schedule import)

##############################
# Schedule registry tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-schedule_registry.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-schedule_registry
  @ONLY)

vistk_declare_test(python-schedule_registry)

vistk_make_python_test(python-schedule_registry import)

##############################
# Stamp tests
##############################
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test-python-stamp.py.in
  ${EXECUTABLE_OUTPUT_PATH}/test-python-stamp
  @ONLY)

vistk_declare_test(python-stamp)

vistk_make_python_test(python-stamp import)
