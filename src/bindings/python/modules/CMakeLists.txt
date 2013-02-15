project(vistk_python_modules)

set(modules_python_srcs
  registration.cxx)

set(modules_python_private_headers
  modules-config.h
  registration.h)

vistk_private_header_group(${modules_python_private_headers})
add_vistk_plugin(modules_python
  MAKE_VISTK_MODULES_PYTHON_LIB
  ${modules_python_srcs}
  ${modules_python_private_headers})
target_link_libraries(modules_python
  LINK_PRIVATE
    vistk_pipeline
    vistk_python_util
    python_exceptions
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

add_python_module(modules.py
  vistk/modules
  modules)

execute_process(
  COMMAND           "${PYTHON_EXECUTABLE}" -c
                    "import pkg_resources; pkg_resources.require('straight.plugin >= 1.3.1')"
  WORKING_DIRECTORY "${vistk_source_dir}"
  RESULT_VARIABLE   straight_return
  OUTPUT_QUIET
  ERROR_QUIET)

set(default_have_straight OFF)

if (NOT straight_return)
  set(default_have_straight ON)
endif ()

option(VISTK_ASSUME_STRAIGHT_PLUGIN "Assume the straight.plugin Python module exists (needs >= 1.3.1)" ${default_have_straight})
if (NOT VISTK_ASSUME_STRAIGHT_PLUGIN)
  add_python_module(loaders.py
    vistk/modules
    loaders)
endif ()

make_python_init_file(vistk/modules)
#  modules)

set (python_both_arch TRUE)

make_python_plugin_init_file(vistk/processes)
make_python_plugin_init_file(vistk/schedulers)
