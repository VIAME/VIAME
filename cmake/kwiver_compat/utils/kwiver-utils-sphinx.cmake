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
  if( ${CMAKE_PROJECT_NAME}_ENABLE_DOCS )
    add_custom_target(sphinx-kwiver
       COMMAND ${SPHINX_EXECUTABLE} -D breathe_projects.kwiver="${CMAKE_BINARY_DIR}/doc/kwiver/xml" ${CMAKE_SOURCE_DIR}/doc/manuals ${CMAKE_BINARY_DIR}/doc/sphinx
    )
    set_target_properties(sphinx-kwiver PROPERTIES FOLDER "Documentation")
    # Our sphinx documentation uses the products of doxygen generation, so let's
    # depend on that.
    add_dependencies( sphinx-kwiver doxygen-kwiver )
  endif( ${CMAKE_PROJECT_NAME}_ENABLE_DOCS )
endfunction(kwiver_create_sphinx)
