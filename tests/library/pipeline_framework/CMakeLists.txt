# pipeline_framework tests

project( test_viame_pipeline_framework )

include( kwiver-test-setup )

###
# The `.pipe` grammar
#
# Recorded before P8-T07 trims the library that parses it. `pipes.json` says
# the 292 shipped pipelines bake; this says what the grammar is, including
# the parts none of them use -- which is what P8-T07 removes on the argument
# that nothing uses them.
##
kwiver_discover_gtests( viame pipe_parser
  LIBRARIES kwiver::vital
            kwiver::vital_config
            sprokit_pipeline
            sprokit_pipeline_util
            GTest::gtest )
