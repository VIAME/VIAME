project(vistk_python_modules)

set(modules_python_srcs
  registration.cxx)

add_vistk_plugin(modules_python
  MAKE_VISTK_MODULES_PYTHON_LIB
  ${modules_python_srcs})
target_link_libraries(modules_python
  vistk_pipeline
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

add_python_module(modules.py
  vistk/modules
  modules)

make_python_init_file(vistk/modules)
#  modules)
make_python_init_file(vistk/processes)
make_python_init_file(vistk/schedules)
