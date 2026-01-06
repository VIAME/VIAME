/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Convert single image to image_set
 */

#include "image_to_image_set_process.h"

#include <sprokit/processes/kwiver_type_traits.h>

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/image_container_set_simple.h>

#include <vector>


namespace viame
{

namespace core
{

namespace kv = kwiver::vital;

//------------------------------------------------------------------------------
// Private implementation class
class image_to_image_set_process::priv
{
public:
  priv();
  ~priv();
};

// =============================================================================

image_to_image_set_process
::image_to_image_set_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new image_to_image_set_process::priv() )
{
  make_ports();
  make_config();
}


image_to_image_set_process
::~image_to_image_set_process()
{
}


// -----------------------------------------------------------------------------
void
image_to_image_set_process
::_configure()
{
}


// -----------------------------------------------------------------------------
void
image_to_image_set_process
::_step()
{
  // Check for completion signal
  auto const& p_info = peek_at_port_using_trait( image );

  if( p_info.datum->type() == sprokit::datum::complete )
  {
    grab_edge_datum_using_trait( image );
    mark_process_as_complete();
    return;
  }

  kv::image_container_sptr img = grab_from_port_using_trait( image );

  // Create image set with single image
  std::vector< kv::image_container_sptr > images;
  if( img )
  {
    images.push_back( img );
  }

  auto image_set = std::make_shared< kv::simple_image_container_set >( images );

  push_to_port_using_trait( image_set, image_set );
}


// -----------------------------------------------------------------------------
void
image_to_image_set_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );

  // -- output --
  declare_output_port_using_trait( image_set, optional );
}


// -----------------------------------------------------------------------------
void
image_to_image_set_process
::make_config()
{
}


// =============================================================================
image_to_image_set_process::priv
::priv()
{
}


image_to_image_set_process::priv
::~priv()
{
}


} // end namespace core

} // end namespace viame
