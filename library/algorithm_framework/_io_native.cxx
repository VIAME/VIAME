// Return the reader's shared pointer by value: Python cannot observe a C++
// replacement of an output shared_ptr passed to ReadObjectTrackSet.read_set.
#include <viame/algorithm_framework/algo/read_object_track_set.h>
#include <viame/utilities/python_fold.h>
#include <pybind11/pybind11.h>

VIAME_PYTHON_MODULE( _io_native, m )
{
  m.def( "read_tracks", []( viame::algo::read_object_track_set& reader )
  {
    auto tracks = std::make_shared< viame::object_track_set >();
    reader.read_set( tracks );
    return tracks;
  } );
}
