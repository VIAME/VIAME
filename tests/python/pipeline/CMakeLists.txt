project(vistk_test_python_pipeline)

##############################
# Config tests
##############################
vistk_build_python_test(config test-python-config.py.in)

vistk_make_python_test(config import)
vistk_make_python_test(config create)

##############################
# Datum tests
##############################
vistk_build_python_test(datum test-python-datum.py.in)

vistk_make_python_test(datum import)

##############################
# Edge tests
##############################
vistk_build_python_test(edge test-python-edge.py.in)

vistk_make_python_test(edge import)

##############################
# Modules tests
##############################
vistk_build_python_test(modules test-python-modules.py.in)

vistk_make_python_test(modules import)

##############################
# Pipeline tests
##############################
vistk_build_python_test(pipeline test-python-pipeline.py.in)

vistk_make_python_test(pipeline import)

##############################
# Process tests
##############################
vistk_build_python_test(process test-python-process.py.in)

vistk_make_python_test(process import)

##############################
# Process registry tests
##############################
vistk_build_python_test(process_registry test-python-process_registry.py.in)

vistk_make_python_test(process_registry import)

##############################
# Schedule tests
##############################
vistk_build_python_test(schedule test-python-schedule.py.in)

vistk_make_python_test(schedule import)

##############################
# Schedule registry tests
##############################
vistk_build_python_test(schedule_registry test-python-schedule_registry.py.in)

vistk_make_python_test(schedule_registry import)

##############################
# Stamp tests
##############################
vistk_build_python_test(stamp test-python-stamp.py.in)

vistk_make_python_test(stamp import)
