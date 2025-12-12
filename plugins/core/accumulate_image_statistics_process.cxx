/**
 * \file
 * \brief Accumulate image statistics (frame count, image dimensions) over a stream
 */

#include "accumulate_image_statistics_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

#include <memory>

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_type_trait( integer, "kwiver:integer", int64_t );
create_port_trait( image_width, integer, "Width of the input images." );
create_port_trait( image_height, integer, "Height of the input images." );

// =============================================================================
// Private implementation class
class accumulate_image_statistics_process::priv
{
public:
  explicit priv( accumulate_image_statistics_process* parent );
  ~priv();

  // Internal variables
  unsigned m_frame_count;
  unsigned m_image_width;
  unsigned m_image_height;
  bool m_image_size_set;

  // Other variables
  accumulate_image_statistics_process* parent;

  kv::logger_handle_t m_logger;
};


// -----------------------------------------------------------------------------
accumulate_image_statistics_process::priv
::priv( accumulate_image_statistics_process* ptr )
  : m_frame_count( 0 )
  , m_image_width( 0 )
  , m_image_height( 0 )
  , m_image_size_set( false )
  , parent( ptr )
  , m_logger( kv::get_logger( "accumulate_image_statistics_process" ) )
{
}


accumulate_image_statistics_process::priv
::~priv()
{
}


// =============================================================================
accumulate_image_statistics_process
::accumulate_image_statistics_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new accumulate_image_statistics_process::priv( this ) )
{
  make_ports();
  make_config();
}


accumulate_image_statistics_process
::~accumulate_image_statistics_process()
{
}


// -----------------------------------------------------------------------------
void
accumulate_image_statistics_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( image, required );

  // -- outputs --
  declare_output_port_using_trait( image_width, optional );
  declare_output_port_using_trait( image_height, optional );
}


// -----------------------------------------------------------------------------
void
accumulate_image_statistics_process
::make_config()
{
  // No configuration parameters needed
}


// -----------------------------------------------------------------------------
void
accumulate_image_statistics_process
::_configure()
{
  // No configuration to read
}

// -----------------------------------------------------------------------------
void
accumulate_image_statistics_process
::_step()
{
  kv::image_container_sptr image;

  image = grab_from_port_using_trait( image );

  // Track image dimensions from the first non-null image
  if( image && !d->m_image_size_set )
  {
    d->m_image_width = static_cast<unsigned>( image->width() );
    d->m_image_height = static_cast<unsigned>( image->height() );
    d->m_image_size_set = true;
    LOG_DEBUG( d->m_logger, "Image size from input: " << d->m_image_width << "x" << d->m_image_height );
  }

  d->m_frame_count++;
  LOG_DEBUG( d->m_logger, "Frame count: " << d->m_frame_count );

  // Check if input is complete
  auto port_info = peek_at_port_using_trait( image );
  auto is_input_complete = port_info.datum->type() == sprokit::datum::complete;

  if( is_input_complete )
  {
    LOG_INFO( d->m_logger, "Input complete. Total frames: " << d->m_frame_count
              << ", Image size: " << d->m_image_width << "x" << d->m_image_height );

    // Push final statistics
    push_to_port_using_trait( image_width, static_cast<int64_t>( d->m_image_width ) );
    push_to_port_using_trait( image_height, static_cast<int64_t>( d->m_image_height ) );

    mark_process_as_complete();
  }
  else
  {
    // Send empty datum while still processing
    const auto dat = sprokit::datum::empty_datum();
    push_datum_to_port_using_trait( image_width, dat );
    push_datum_to_port_using_trait( image_height, dat );
  }
}

} // end namespace core

} // end namespace viame
