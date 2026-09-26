// Return the reader's shared pointer by value: Python cannot observe a C++
// replacement of an output shared_ptr passed to ReadObjectTrackSet.read_set.
#include <vital/algo/read_object_track_set.h>
#include <pybind11/pybind11.h>
#include <sprokit/processes/adapters/adapter_data_set.h>

PYBIND11_MODULE( _io_native, m )
{
  // AdapterDataSet's generic Python float conversion stores a C++ float;
  // video input frame_rate ports require a double.
  m.def( "add_double", []( kwiver::adapter::adapter_data_set& data,
                          std::string const& port, double value )
  {
    data.add_value<double>( port, value );
  } );
  m.def( "get_double", []( kwiver::adapter::adapter_data_set& data,
                          std::string const& port )
  {
    return data.value<double>( port );
  } );
  m.def( "read_tracks", []( kwiver::vital::algo::read_object_track_set& reader )
  {
    auto tracks = std::make_shared< kwiver::vital::object_track_set >();
    reader.read_set( tracks );
    return tracks;
  } );
}
