set( json_python_headers
     serialize_activity.h
     serialize_activity_type.h
     serialize_bounding_box.h
     serialize_detected_object.h
     serialize_detected_object_type.h
     serialize_detected_object_set.h
     serialize_image.h
     serialize_timestamp.h
     serialize_track_state.h
     serialize_track.h
     serialize_track_set.h
     serialize_object_track_state.h
     serialize_object_track_set.h
   )

set( json_python_sources
     serialize_utils.txx
     serialize_activity.cxx
     serialize_activity_type.cxx
     serialize_bounding_box.cxx
     serialize_detected_object.cxx
     serialize_detected_object_type.cxx
     serialize_detected_object_set.cxx
     serialize_image.cxx
     serialize_module.cxx
     serialize_timestamp.cxx
     serialize_track_state.cxx
     serialize_track.cxx
     serialize_track_set.cxx
     serialize_object_track_state.cxx
     serialize_object_track_set.cxx
   )

kwiver_add_python_library( json
   arrows/serialize/json
   ${json_python_headers}
   ${json_python_sources}
)

target_link_libraries( python-arrows.serialize.json-json
  PUBLIC ${PYTHON_LIBRARIES}
         vital
         kwiver_serialize_json
)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                                arrows/serialize/json
                                __init__)
