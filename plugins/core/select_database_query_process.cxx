/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Select between two database query inputs
 */

#include "select_database_query_process.h"

#include <sprokit/processes/kwiver_type_traits.h>

#include <vital/vital_types.h>
#include <vital/types/database_query.h>


namespace viame
{

namespace core
{

namespace kv = kwiver::vital;

create_port_trait( primary_query, database_query,
  "Primary database query input - used if non-null" );
create_port_trait( fallback_query, database_query,
  "Fallback database query input - used if primary is null" );
create_port_trait( use_primary, bool,
  "Flag to use primary and skip fallback - typically connected to boxes_provided" );

//------------------------------------------------------------------------------
// Private implementation class
class select_database_query_process::priv
{
public:
  priv();
  ~priv();
};

// =============================================================================

select_database_query_process
::select_database_query_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new select_database_query_process::priv() )
{
  make_ports();
  make_config();
}


select_database_query_process
::~select_database_query_process()
{
}


// -----------------------------------------------------------------------------
void
select_database_query_process
::_configure()
{
}


// -----------------------------------------------------------------------------
void
select_database_query_process
::_step()
{
  // Check for completion signal
  auto const& p_info = peek_at_port_using_trait( primary_query );

  if( p_info.datum->type() == sprokit::datum::complete )
  {
    grab_edge_datum_using_trait( primary_query );
    if( has_input_port_edge_using_trait( use_primary ) )
    {
      grab_edge_datum_using_trait( use_primary );
    }
    if( has_input_port_edge_using_trait( fallback_query ) )
    {
      grab_edge_datum_using_trait( fallback_query );
    }
    mark_process_as_complete();
    return;
  }

  kv::database_query_sptr primary;
  kv::database_query_sptr fallback;

  primary = grab_from_port_using_trait( primary_query );

  // Check if use_primary flag is set - when true, skip waiting for fallback
  bool skip_fallback = false;
  if( has_input_port_edge_using_trait( use_primary ) )
  {
    skip_fallback = grab_from_port_using_trait( use_primary );
  }

  // Only grab from fallback if we're not skipping it and it's connected
  if( !skip_fallback && has_input_port_edge_using_trait( fallback_query ) )
  {
    fallback = grab_from_port_using_trait( fallback_query );
  }

  // Select primary if non-null, otherwise use fallback
  if( primary )
  {
    push_to_port_using_trait( database_query, primary );
  }
  else
  {
    push_to_port_using_trait( database_query, fallback );
  }
}


// -----------------------------------------------------------------------------
void
select_database_query_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( primary_query, required );
  declare_input_port_using_trait( fallback_query, optional );
  declare_input_port_using_trait( use_primary, optional );

  // -- output --
  declare_output_port_using_trait( database_query, optional );
}


// -----------------------------------------------------------------------------
void
select_database_query_process
::make_config()
{
}


// =============================================================================
select_database_query_process::priv
::priv()
{
}


select_database_query_process::priv
::~priv()
{
}


} // end namespace core

} // end namespace viame
