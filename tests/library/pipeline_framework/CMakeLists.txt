# pipeline_framework tests

project( test_viame_pipeline_framework )

include( viame-test-setup )

###
# The `.pipe` grammar
#
# Recorded before P8-T07 trims the library that parses it. `pipes.json` says
# the 292 shipped pipelines bake; this says what the grammar is, including
# the parts none of them use -- which is what P8-T07 removes on the argument
# that nothing uses them.
##
viame_discover_gtests( viame pipe_parser
  LIBRARIES kwiver::vital
            kwiver::vital_config
            viame_pipeline_framework
            viame_pipeline_util
            GTest::gtest )

###
# The C++ schedulers
#
# Every shipped pipeline names `pythread_per_process`, so `thread_per_process`
# is reached only as the default and `sync` not at all. 292 green pipelines
# say nothing about either, which is why they need a test before P8-T07
# removes one of them.
##
viame_discover_gtests( viame scheduler
  LIBRARIES kwiver::vital
            kwiver::vital_config
            kwiver::vital_vpm
            viame::registry
            viame_pipeline_framework
            viame_pipeline_util
            GTest::gtest )

###
# Process type aliases
#
# Added by P8-T07 so that a renamed process is a rename in the compatibility
# baseline rather than a removal and an addition.
##
viame_discover_gtests( viame process_alias
  LIBRARIES kwiver::vital
            kwiver::vital_config
            kwiver::vital_vpm
            viame::registry
            viame_pipeline_framework
            viame_pipeline_util
            GTest::gtest )
