# VIAME Tests
#
# What is no one library's: the compatibility baseline, the startup budget,
# the golden recordings, the tools, and the examples and pipelines. Unit tests
# live beside the code they test, in `library/<dir>/tests`, since P2-T09.
# Tests are only built when VIAME_ENABLE_TESTS is ON.

project( viame_tests )

include( viame-test-setup )

add_subdirectory( baseline )
add_subdirectory( startup )
add_subdirectory( golden )
add_subdirectory( tools )
add_subdirectory( examples )
add_subdirectory( pipelines )
