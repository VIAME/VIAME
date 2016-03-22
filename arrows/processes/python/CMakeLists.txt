#
# KWIVER processes
#

# Sprokit macros & stuff
include("${sprokit_source_dir}/conf/sprokit-macro-targets.cmake")
include("${sprokit_source_dir}/conf/sprokit-macro-configure.cmake")
include("${sprokit_source_dir}/conf/sprokit-macro-python.cmake")

set(sprokit_python_subdir "python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")
set(sprokit_python_output_path "${KWIVER_BUILD_INSTALL_PREFIX}/lib/${sprokit_python_subdir}")

sprokit_create_python_plugin_init(kwiver)
sprokit_create_python_plugin_init(kwiver/processes)

# add python processes here
sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/kw_print_number_process.py
  kwiver/processes
  kw_print_number_process)

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/homography_writer.py
  kwiver/processes
  homography_writer)

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/ProcessImage.py
  kwiver/processes
  ProcessImage)

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/ApplyDescriptor.py
  kwiver/processes
  ApplyDescriptor)
