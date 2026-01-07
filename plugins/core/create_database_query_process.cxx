/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Create a database query from track descriptors
 */

#include "create_database_query_process.h"

#include <sprokit/processes/kwiver_type_traits.h>

#include <vital/vital_types.h>

#include <vital/types/database_query.h>
#include <vital/types/track_descriptor_set.h>
#include <vital/types/uid.h>

#include <sstream>


namespace viame
{

namespace core
{

namespace kv = kwiver::vital;

create_config_trait( query_type, std::string, "similarity",
  "Type of query: 'similarity' or 'retrieval'" );

create_config_trait( threshold, double, "0.0",
  "Relevancy threshold for query results" );

create_port_trait( boxes_provided, bool,
  "Flag indicating if bounding boxes were provided - only create query when true" );

//------------------------------------------------------------------------------
// Private implementation class
class create_database_query_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  kv::database_query::query_type m_query_type;
  double m_threshold;

  // Query ID counter
  unsigned m_query_counter;
};

// =============================================================================

create_database_query_process
::create_database_query_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new create_database_query_process::priv() )
{
  make_ports();
  make_config();
}


create_database_query_process
::~create_database_query_process()
{
}


// -----------------------------------------------------------------------------
void
create_database_query_process
::_configure()
{
  std::string type_str = config_value_using_trait( query_type );

  if( type_str == "similarity" )
  {
    d->m_query_type = kv::database_query::SIMILARITY;
  }
  else if( type_str == "retrieval" )
  {
    d->m_query_type = kv::database_query::RETRIEVAL;
  }
  else
  {
    d->m_query_type = kv::database_query::SIMILARITY;
  }

  d->m_threshold = config_value_using_trait( threshold );
}


// -----------------------------------------------------------------------------
void
create_database_query_process
::_step()
{
  // Check for completion signal
  auto const& p_info = peek_at_port_using_trait( track_descriptor_set );

  if( p_info.datum->type() == sprokit::datum::complete )
  {
    grab_edge_datum_using_trait( track_descriptor_set );
    if( has_input_port_edge_using_trait( boxes_provided ) )
    {
      grab_edge_datum_using_trait( boxes_provided );
    }
    mark_process_as_complete();
    return;
  }

  kv::track_descriptor_set_sptr descriptors;
  descriptors = grab_from_port_using_trait( track_descriptor_set );

  // Check if boxes_provided flag is connected and if boxes were provided
  bool should_create_query = true;
  if( has_input_port_edge_using_trait( boxes_provided ) )
  {
    should_create_query = grab_from_port_using_trait( boxes_provided );
  }

  // Only create a query when boxes were provided (for auto-query mode)
  // When no boxes, output null so perform_query waits for external input
  if( !should_create_query )
  {
    push_to_port_using_trait( database_query, kv::database_query_sptr() );
    return;
  }

  // Create new database query
  auto query = std::make_shared< kv::database_query >();

  // Generate unique ID for this query
  std::stringstream ss;
  ss << "query_" << d->m_query_counter++;
  query->set_id( kv::uid( ss.str() ) );

  // Set query type
  query->set_type( d->m_query_type );

  // Set descriptors
  if( descriptors )
  {
    query->set_descriptors( descriptors );
  }

  // Set threshold
  query->set_threshold( d->m_threshold );

  push_to_port_using_trait( database_query, query );
}


// -----------------------------------------------------------------------------
void
create_database_query_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( track_descriptor_set, required );
  declare_input_port_using_trait( boxes_provided, optional );

  // -- output --
  declare_output_port_using_trait( database_query, optional );
}


// -----------------------------------------------------------------------------
void
create_database_query_process
::make_config()
{
  declare_config_using_trait( query_type );
  declare_config_using_trait( threshold );
}


// =============================================================================
create_database_query_process::priv
::priv()
  : m_query_type( kv::database_query::SIMILARITY )
  , m_threshold( 0.0 )
  , m_query_counter( 0 )
{
}


create_database_query_process::priv
::~priv()
{
}


} // end namespace core

} // end namespace viame
