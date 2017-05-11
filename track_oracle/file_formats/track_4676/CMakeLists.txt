#
# STANAG 4676 tracks
#

add_library( track_4676
             track_4676.h
             file_format_4676.h    file_format_4676.cxx )
set_target_properties( track_4676 PROPERTIES
  VERSION ${vidtk_VERSION} SOVERSION ${vidtk_VERSION}
)

target_link_libraries( track_4676
  PUBLIC track_oracle track_oracle_format_base
  PRIVATE track_oracle_tokenizers vidtk_stanag_4676 vidtk_logger vgl ${TinyXML_LIBRARY}
  )
install( TARGETS track_4676
  RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib
)
