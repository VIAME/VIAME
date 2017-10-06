
ExternalData_Add_Test(external_darknet_example
  NAME Download
  COMMAND 
  DATA{darknet.zip}
  )
ExternalData_Add_Target(external_darknet_example)

# Make it so user needs to manually pull the data
set_target_properties(external_darknet_example PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)

add_custom_target(setup_darknet_example)
add_dependencies(setup_darknet_example external_darknet_example)
add_custom_command(TARGET setup_darknet_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${EXAMPLE_DIR}/external/darknet.zip 
            WORKING_DIRECTORY ${EXAMPLE_DIR}/pipelines)
add_custom_command(TARGET setup_darknet_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DEXAMPLE_DIR:STRING=${EXAMPLE_DIR} -P configure.cmake 
            WORKING_DIRECTORY ${EXAMPLE_DIR}/pipelines/darknet)