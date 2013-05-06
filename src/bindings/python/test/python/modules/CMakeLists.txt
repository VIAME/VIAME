project(sprokit_python_test_python_modules)

add_python_module(extras.py
  sprokit/test/python/modules
  extras)

make_python_plugin_init_file(sprokit/test/python/modules)
