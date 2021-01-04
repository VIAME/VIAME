// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "downsample_process.h"

#include <kwiver_type_traits.h>

#include <vital/types/timestamp.h>
#include <vital/vital_config.h>

namespace kwiver
{

using sprokit::process;

create_config_trait( target_frame_rate, double, "1.0", "Target frame rate" );
create_config_trait( burst_frame_count, unsigned, "0", "Burst frame count" );
create_config_trait( burst_frame_break, unsigned, "0", "Burst frame break" );
create_config_trait( renumber_frames, bool, "false", "Renumber output frames" );

class downsample_process::priv
{
public:
  explicit priv( downsample_process* p );
  ~priv();

  bool skip_frame( VITAL_UNUSED vital::timestamp const& ts, double frame_rate );

  downsample_process* parent;

  double target_frame_rate_;
  unsigned burst_frame_count_;
  unsigned burst_frame_break_;
  bool renumber_frames_;

  // Time of the current frame (seconds)
  double ds_frame_time_;
  // Time of the last sent frame (ignoring burst filtering)
  double last_sent_frame_time_;
  unsigned burst_counter_;
  unsigned output_counter_;
  bool is_first_;

  static port_t const port_inputs[5];
  static port_t const port_outputs[5];

private:
  // Compute the frame number corresponding to time_seconds assuming a
  // frame rate of target_frame_rate_
  int target_frame_count( double time_seconds );
};

process::port_t const downsample_process::priv::port_inputs[5] = {
  process::port_t( "input_1" ),
  process::port_t( "input_2" ),
  process::port_t( "input_3" ),
  process::port_t( "input_4" ),
  process::port_t( "input_5" ),
};

process::port_t const downsample_process::priv::port_outputs[5] = {
  process::port_t( "output_1" ),
  process::port_t( "output_2" ),
  process::port_t( "output_3" ),
  process::port_t( "output_4" ),
  process::port_t( "output_5" ),
};

downsample_process
::downsample_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new downsample_process::priv( this ) )
{
  make_ports();
  make_config();
}

downsample_process
::~downsample_process()
{
}

void downsample_process
::_configure()
{
  d->target_frame_rate_ = config_value_using_trait( target_frame_rate );
  d->burst_frame_count_ = config_value_using_trait( burst_frame_count );
  d->burst_frame_break_ = config_value_using_trait( burst_frame_break );
  d->renumber_frames_ = config_value_using_trait( renumber_frames );
}

void downsample_process
::_init()
{
  d->ds_frame_time_ = 0.0;
  d->last_sent_frame_time_ = 0.0;
  d->burst_counter_ = 0;
  d->output_counter_ = 0;
  d->is_first_ = true;
}

void downsample_process
::_step()
{
  kwiver::vital::timestamp ts = grab_from_port_using_trait( timestamp );
  double frame_rate = grab_from_port_using_trait( frame_rate );
  bool send_frame = !d->skip_frame( ts, frame_rate );

  if( send_frame )
  {
    if( d->renumber_frames_ )
    {
      ts.set_frame( d->output_counter_++ );
    }

    LOG_DEBUG( logger(), "Sending frame " << ts.get_frame() );
    push_to_port_using_trait( timestamp, ts );
  }

  for( size_t i = 0; i < 5; i++ )
  {
    if( has_input_port_edge( d->port_inputs[i] ) )
    {
      sprokit::datum_t datum = grab_datum_from_port( d->port_inputs[i] );
      if( send_frame )
      {
        push_datum_to_port( d->port_outputs[i], datum );
      }
    }
  }
}

void downsample_process
::make_ports()
{
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( frame_rate, required );
  for( size_t i = 0; i < 5; i++ )
  {
    declare_input_port( priv::port_inputs[i],
                        type_any,
                        optional,
                        port_description_t( "Input data." ) );
  }

  declare_output_port_using_trait( timestamp, optional );
  for( size_t i = 0; i < 5; i++ )
  {
    declare_output_port( priv::port_outputs[i],
                         type_any,
                         optional,
                         port_description_t( "Output data." ) );
  }
}

void downsample_process
::make_config()
{
  declare_config_using_trait( target_frame_rate );
  declare_config_using_trait( burst_frame_count );
  declare_config_using_trait( burst_frame_break );
  declare_config_using_trait( renumber_frames );
}

int downsample_process::priv
::target_frame_count( double time_seconds )
{
  return static_cast< int >( std::floor( time_seconds * target_frame_rate_ ) );
}

bool downsample_process::priv
::skip_frame( VITAL_UNUSED vital::timestamp const& ts,
              double frame_rate )
{
  ds_frame_time_ = ts.has_valid_time() ?
    ts.get_time_seconds() : ds_frame_time_ + 1 / frame_rate;

  if( is_first_ )
  {
    // Triggers always sending the first frame
    last_sent_frame_time_ = ( target_frame_count( ds_frame_time_ ) - 0.5 ) / target_frame_rate_;
    is_first_ = false;
  }

  int elapsed_frames = target_frame_count( ds_frame_time_ )
    - target_frame_count( last_sent_frame_time_ );
  if( elapsed_frames <= 0 )
  {
    return true;
  }
  else
  {
    last_sent_frame_time_ = ds_frame_time_;
  }

  if( burst_frame_count_ != 0 && burst_frame_break_ != 0 )
  {
    burst_counter_ += elapsed_frames;
    burst_counter_ %= burst_frame_count_ + burst_frame_break_;

    // If burst_counter_ is in [1..burst_frame_count_], we're in
    // pass-through mode; otherwise we're in skip mode.
    if( burst_counter_ > burst_frame_count_ || burst_counter_ == 0)
    {
      return true;
    }
  }

  return false;
}

downsample_process::priv
::priv( downsample_process* p )
  : parent( p )
{
}

downsample_process::priv
::~priv()
{
}

}
