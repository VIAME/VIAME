###
# Text query and VLM examples
##

set( EXAMPLE_INSTALL_FILES
     README.rst
     input_list.txt )

if( WIN32 )
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_sam3_text_query.bat
       run_ollama_vlm_text_query.bat )
else()
  set( EXAMPLE_INSTALL_FILES
       ${EXAMPLE_INSTALL_FILES}
       run_sam3_text_query.sh
       run_ollama_vlm_text_query.sh )
endif()

install( FILES       ${EXAMPLE_INSTALL_FILES}
         DESTINATION examples/text_query_and_vlm )
