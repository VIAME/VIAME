###
# Conversion scripts for type conversion
##

set( example_files
  annotation_converter.pl
  habcam_to_kw18.pl
  threshold_kw18_bbox.pl
  scallop-tk_to_kw18.pl
  detection_set_to_kw18.pl
  scallop-tk_to_pvo.pl )

install( FILES       ${example_files}
         DESTINATION examples/detection_file_conversions )
