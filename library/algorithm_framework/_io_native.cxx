// Return the reader's shared pointer by value: Python cannot observe a C++
// replacement of an output shared_ptr passed to ReadObjectTrackSet.read_set.
#include <viame/algorithm_framework/algo/read_object_track_set.h>
#include <viame/utilities/python_fold.h>
#include <pybind11/pybind11.h>
#include <viame/pipeline_framework/adapters/adapter_data_set.h>

VIAME_PYTHON_MODULE( _io_native, m )
{
  // AdapterDataSet's generic Python float conversion stores a C++ float;
  // video input frame_rate ports require a double.
  m.def( "add_double", []( viame::adapter::adapter_data_set& data,
                          std::string const& port, double value )
  {
    data.add_value<double>( port, value );
  } );
  m.def( "get_double", []( viame::adapter::adapter_data_set& data,
                          std::string const& port )
  {
    return data.value<double>( port );
  } );
  m.def( "read_tracks", []( viame::algo::read_object_track_set& reader )
  {
    auto tracks = std::make_shared< viame::object_track_set >();
    reader.read_set( tracks );
    return tracks;
  } );
}
