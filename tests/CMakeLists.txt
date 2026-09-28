# VIAME Tests
#
# Library unit tests mirror library/<component> in tests/library/<component>.
# Baselines, golden recordings, tools, examples and pipelines also live here.
# Tests are only built when VIAME_ENABLE_TESTS is ON.

project( viame_tests )

include( viame-test-setup )

add_subdirectory( library )
add_subdirectory( baseline )
add_subdirectory( startup )
add_subdirectory( reference )
add_subdirectory( tools )
add_subdirectory( examples )
add_subdirectory( pipelines )
