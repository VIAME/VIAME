// Return the reader's shared pointer by value: Python cannot observe a C++
// replacement of an output shared_ptr passed to ReadObjectTrackSet.read_set.
#include <vital/algo/read_object_track_set.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE( _io_native, m )
{
  m.def( "read_tracks", []( kwiver::vital::algo::read_object_track_set& reader )
  {
    auto tracks = std::make_shared< kwiver::vital::object_track_set >();
    reader.read_set( tracks );
    return tracks;
  } );
}
