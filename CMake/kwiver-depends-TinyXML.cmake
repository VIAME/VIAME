# if track_oracle is enabled, need to find TinyXML

if (KWIVER_ENABLE_TRACK_ORACLE)
  find_package( TinyXML REQUIRED )
  add_definitions( -DTIXML_USE_STL )
  include_directories( SYSTEM ${TinyXML_INCLUDE_DIR} )
endif (KWIVER_ENABLE_TRACK_ORACLE)