#
# Python converters
#
# This cmake file generates the conversion libraries needed to access
# kwiver and assiclated types.
#
# The naming convention is kwiver_python_convert_<c++_class_name>

project( core_conversions )

include_directories( SYSTEM ${PYTHON_INCLUDE_DIR} )

# need access to sprokit private includes
include_directories( ${KWIVER_PACKAGES_DIR}/sprokit/src )

# ==========================================
add_library( kwiver_python_convert_homography
  homography.cxx)

target_link_libraries( kwiver_python_convert_homography
  LINK_PRIVATE
  sprokit_python_any_conversion
  sprokit_python_util
  maptk_core
  boost_python
 # ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_dependencies( kwiver_python_convert_homography
  sprokit maptk VXL )


#add_library( kwiver_python_convert_timestamp
#  timestamp.cxx)



# make_python_init_file(core/conversions)
####
# INSTALL
#
install( TARGETS kwiver_python_convert_homography
  DESTINATION ${KWIVER_BUILD_INSTALL_PREFIX}/lib/python/site-packages/kwiver/processes
  )
