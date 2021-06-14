#
# Setup and define KWIVER sphinx support
#
find_package(Sphinx)

include(CMakeDependentOption)
cmake_dependent_option(${CMAKE_PROJECT_NAME}_ENABLE_SPHINX_DOCS
  "Build KWIVER documentation via Sphinx." OFF
  SPHINX_FOUND OFF
  )

function(kwiver_create_sphinx)
   add_custom_target(sphinx-kwiver
     COMMAND ${SPHINX_EXECUTABLE} -D breathe_projects.kwiver="${CMAKE_BINARY_DIR}/doc/kwiver/xml" ${CMAKE_SOURCE_DIR}/doc/manuals ${CMAKE_BINARY_DIR}/doc/sphinx
  )
  set_target_properties(sphinx-kwiver PROPERTIES FOLDER "Documentation")

endfunction(kwiver_create_sphinx)
