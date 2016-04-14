#
# Top level for arrows
#
add_subdirectory( algorithms )

if (ARROWS_ENABLE_SPROKIT)
  add_subdirectory( pipelines )
  add_subdirectory( processes )
endif()
