# The pipeline framework
#
# The framework itself is still built by kwiver's sprokit CMakeLists, which
# name these sources through VIAME_LITE_LIBRARY_DIR; P5-T05 is where that
# moves here. What is here now is the process plugin, which has no counterpart
# left in kwiver.

add_subdirectory( processes )
