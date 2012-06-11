project(vistk_python_modules)

set(modules_python_srcs
  registration.cxx)

add_vistk_plugin(modules_python
  MAKE_VISTK_MODULES_PYTHON_LIB
  ${modules_python_srcs})
target_link_libraries(modules_python
  vistk_pipeline
  vistk_python_util
  python_exceptions
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_python_module(modules.py
  vistk/modules
  modules)

option(VISTK_ASSUME_STRAIGHT_PLUGIN "Assume the straight.plugin Python module exists" OFF)
if (NOT VISTK_ASSUME_STRAIGHT_PLUGIN)
  add_python_module(loaders.py
    vistk/modules
    loaders)
endif (NOT VISTK_ASSUME_STRAIGHT_PLUGIN)

make_python_init_file(vistk/modules)
#  modules)

set (python_both_arch TRUE)

make_python_plugin_init_file(vistk/processes)
make_python_plugin_init_file(vistk/schedulers)
