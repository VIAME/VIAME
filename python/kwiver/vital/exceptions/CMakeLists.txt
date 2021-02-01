#
# Python support
#

# Add python modules
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.py
  vital/exceptions
  algorithm )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/base.py
  vital/exceptions
  base )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/config_block.py
  vital/exceptions
  config_block )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/config_block_io.py
  vital/exceptions
  config_block_io )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/eigen.py
  vital/exceptions
  eigen )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/image.py
  vital/exceptions
  image )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/math.py
  vital/exceptions
  math )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  vital/exceptions
  __init__ )
