#
# Python support
#

# Add python modules
viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.py
  vital/exceptions
  algorithm )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/base.py
  vital/exceptions
  base )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/config_block.py
  vital/exceptions
  config_block )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/config_block_io.py
  vital/exceptions
  config_block_io )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/eigen.py
  vital/exceptions
  eigen )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/image.py
  vital/exceptions
  image )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/math.py
  vital/exceptions
  math )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  vital/exceptions
  __init__ )
