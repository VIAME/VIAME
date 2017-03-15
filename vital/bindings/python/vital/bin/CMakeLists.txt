#
# Python support
#

## kwiver_create_python_plugin_init(vital/bin)

# Add python modules
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/maptk_track_features.py
  vital/bin
  maptk_track_features )
