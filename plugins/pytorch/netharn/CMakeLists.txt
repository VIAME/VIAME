# Netharn Training Framework
#
# This directory contains the merged netharn and bioharn packages for
# VIAME's PyTorch-based training and detection functionality.

# Helper function to recursively add all Python modules in a directory
function(add_python_modules_recursive base_path install_prefix)
  file(GLOB_RECURSE _python_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
       "${base_path}/*.py")

  foreach(_py_file ${_python_files})
    # Get the directory and filename
    get_filename_component(_py_dir ${_py_file} DIRECTORY)
    get_filename_component(_py_name ${_py_file} NAME_WE)

    # Build the module path
    if(_py_dir)
      set(_mod_path "${install_prefix}/${_py_dir}")
    else()
      set(_mod_path "${install_prefix}")
    endif()

    kwiver_add_python_module(
      ${CMAKE_CURRENT_SOURCE_DIR}/${_py_file}
      ${_mod_path}
      ${_py_name}
    )
  endforeach()
endfunction()

# Create init file for netharn package
kwiver_create_python_init( arrows/pytorch/netharn )

# Add all Python modules from netharn directory
add_python_modules_recursive( "." "arrows/pytorch/netharn" )
