#
# example using the track oracle
#

add_executable( track_oracle_example
  track_oracle_example.cxx
)

target_link_libraries( track_oracle_example
  track_oracle
  data_terms
  vital
  vital_logger
)

#
# example track reader
#

add_executable( track_reader_example
  track_reader_example.cxx
)

target_link_libraries( track_reader_example
 track_oracle
 data_terms
 track_oracle_file_formats
)

if( KWIVER_ENABLE_KPF )
  target_link_libraries( track_reader_example
    kpf_utils
  )
endif()

#
# example track writer
#

add_executable( track_writer_example
 track_writer_example.cxx )

target_link_libraries( track_writer_example
 track_oracle
 track_oracle_file_formats
)

#
# example track schema introspection
#

add_executable( track_format_example
  track_format_example.cxx
)

target_link_libraries( track_format_example
 track_oracle
 track_oracle_file_formats
)

#
# example user-specified field manipulation
#

add_executable( runtime_field_query_example
 runtime_field_query_example.cxx
)

target_link_libraries( runtime_field_query_example
 track_oracle
 track_oracle_file_formats
)

#
# test the xml tokenizer
#

add_executable( xml_tokenizer_example
  xml_tokenizer_example.cxx
)
target_link_libraries( xml_tokenizer_example
  vital_logger
  track_oracle_tokenizers
)

#
# example KPF activity parser
#

if( KWIVER_ENABLE_KPF )

  add_executable( kpf_activity_example
    kpf_activity_example.cxx
  )

  target_link_libraries( kpf_activity_example
    vital_logger
    track_oracle
    data_terms
    track_oracle_file_formats
  )
endif()
