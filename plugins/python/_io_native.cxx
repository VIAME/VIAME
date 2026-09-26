// Return the reader's shared pointer by value: Python cannot observe a C++
// replacement of an output shared_ptr passed to ReadObjectTrackSet.read_set.
#include <vital/algo/read_object_track_set.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sprokit/pipeline_util/pipeline_builder.h>
#include <sprokit/processes/adapters/adapter_data_set.h>

PYBIND11_MODULE( _io_native, m )
{
  // Parse with the native grammar: includes, environment substitutions and
  // relativepath values retain the semantics of the normal pipeline runner.
  m.def( "pipeline_description", []( std::string const& path,
                                    std::string const& search_path )
  {
    sprokit::pipeline_builder builder;
    builder.add_search_path( search_path );
    builder.load_pipeline( path );
    pybind11::dict processes, config;
    pybind11::list connections;
    for( auto const& block : builder.pipeline_blocks() )
    {
      if( auto const* proc = std::get_if<sprokit::process_pipe_block>( &block ) )
      {
        if( processes.contains( proc->name.c_str() ) )
          throw std::runtime_error( "Duplicate process: " + proc->name );
        processes[proc->name.c_str()] = proc->type;
      }
      else if( auto const* edge = std::get_if<sprokit::connect_pipe_block>( &block ) )
        connections.append( pybind11::make_tuple( edge->from, edge->to ) );
    }
    auto cfg = builder.config();
    for( auto const& key : cfg->available_values() )
      config[key.c_str()] = cfg->get_value<std::string>( key );
    return pybind11::make_tuple( processes, connections, config );
  } );
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
