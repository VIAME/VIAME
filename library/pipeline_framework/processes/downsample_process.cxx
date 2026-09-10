// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "downsample_process.h"

#include <viame/pipeline_framework/type_traits.h>

#include <viame/core_types/timestamp.h>
#include <viame/algorithm_framework/util/string.h>
#include <viame/algorithm_framework/vital_config.h>

namespace kwiver
{

using sprokit::process;

create_config_trait( target_frame_rate, double, "-1.0", "Target frame rate" );
create_config_trait( burst_frame_count, unsigned, "0", "Burst frame count" );
create_config_trait( burst_frame_break, unsigned, "0", "Burst frame break" );
create_config_trait( renumber_frames, bool, "false", "Renumber output frames" );
create_config_trait( only_frames_with_dets, bool, "false", "Frames with dets only" );
create_config_trait( start_time, std::string, "", "Start time to pass frames" );
create_config_trait( duration, std::string, "", "Maximum duration time" );
create_config_trait( start_frame, int, "-1",
  "First frame to include in output. -1 disables. Interpretation depends on frame_range_is_native." );
create_config_trait( end_frame, int, "-1",
  "Last frame to include in output. -1 disables. Interpretation depends on frame_range_is_native." );
create_config_trait( frame_range_is_native, bool, "false",
  "If false (default), start_frame and end_frame are in the downsampled output frame "
  "space (0-indexed count of frames that pass the downsample filter). If true, they "
  "refer to the native input frame numbers." );
create_config_trait( adjust_timestamps, bool, "false",
  "If true, output frame numbers restart from 0 and timestamps are offset so the "
  "first output frame has time 0. If false, original values are preserved "
  "(subject to renumber_frames)." );

create_port_trait( original_timestamp, timestamp, "Timestamp output" );

class downsample_process::priv
{
public:
  explicit priv( downsample_process* p );
  ~priv();

  bool skip_frame( vital::timestamp const& ts, double frame_rate );

  downsample_process* parent;

  typedef vital::timestamp::frame_t frame_t;

  double target_frame_rate_;
  unsigned burst_frame_count_;
  unsigned burst_frame_break_;
  bool renumber_frames_;
  bool only_frames_with_dets_;
  double start_time_;
  double duration_;

  int start_frame_;
  int end_frame_;
  bool frame_range_is_native_;
  bool adjust_timestamps_;

  int64_t ds_send_counter_;
  double first_output_time_;
  bool first_output_seen_;

  // Buffer for old to new frame ids
  std::map< frame_t, frame_t > frame_id_map_;
  // Time of the current frame (seconds)
  double ds_frame_time_;
  // Time of the last sent frame (ignoring burst filtering)
  double last_sent_frame_time_;
  unsigned burst_counter_;
  unsigned output_counter_;
  bool is_first_;

  static port_t const port_inputs[5];
  static port_t const port_outputs[5];

  // Adjust track IDs if timestamps get renumbered from the input to
  // the output if the input datum is an object track set
  sprokit::datum_t adjust_track_ids( const sprokit::datum_t& input );

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
  attach_logger( vital::get_logger( name() ) );

  set_data_checking_level( check_sync );

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
  d->only_frames_with_dets_ = config_value_using_trait( only_frames_with_dets );

  std::string start_time_str = config_value_using_trait( start_time );
  std::string duration_str = config_value_using_trait( duration );

  if( !start_time_str.empty() )
  {
    d->start_time_ = vital::time_str_to_seconds( start_time_str );
  }
  if( !duration_str.empty() )
  {
    d->duration_ = vital::time_str_to_seconds( duration_str );
  }
  if( d->duration_ > 0.0 && d->start_time_ < 0.0 )
  {
    d->start_time_ = 0.0;
  }

  d->start_frame_ = config_value_using_trait( start_frame );
  d->end_frame_ = config_value_using_trait( end_frame );
  d->frame_range_is_native_ = config_value_using_trait( frame_range_is_native );
  d->adjust_timestamps_ = config_value_using_trait( adjust_timestamps );
}

void downsample_process
::_init()
{
  d->ds_frame_time_ = 0.0;
  d->last_sent_frame_time_ = 0.0;
  d->burst_counter_ = 0;
  d->output_counter_ = 0;
  d->is_first_ = true;
  d->frame_id_map_.clear();
  d->ds_send_counter_ = 0;
  d->first_output_time_ = 0.0;
  d->first_output_seen_ = false;
}

void downsample_process
::_step()
{
  bool is_finished = false;
  bool send_frame = true;

  kwiver::vital::timestamp orig_ts, ts;
  double frame_rate = -1.0;

  if( has_input_port_edge_using_trait( timestamp ) )
  {
    auto port_info = peek_at_port_using_trait( timestamp );

    if( port_info.datum->type() == sprokit::datum::complete )
    {
      grab_edge_datum_using_trait( timestamp );
      is_finished = true;
    }
    else
    {
      ts = grab_from_port_using_trait( timestamp );
      orig_ts = ts;
    }
  }

  if( has_input_port_edge_using_trait( frame_rate ) )
  {
    auto port_info = peek_at_port_using_trait( frame_rate );

    if( port_info.datum->type() == sprokit::datum::complete )
    {
      grab_edge_datum_using_trait( frame_rate );
      is_finished = true;
    }
    else
    {
      frame_rate = grab_from_port_using_trait( frame_rate );
    }
  }

  if( d->is_first_ && process::count_output_port_edges( "frame_rate" ) > 0 )
  {
    // With downsampling off every frame is passed through, so the rate
    // downstream sees is the source's own. Reporting the disabled target
    // instead hands encoders a negative rate they cannot open with.
    push_to_port_using_trait( frame_rate,
      d->target_frame_rate_ > 0.0 ? d->target_frame_rate_ : frame_rate );
    push_datum_to_port_using_trait( frame_rate, sprokit::datum::complete_datum() );
  }

  if( d->target_frame_rate_ > 0.0 )
  {
    if( frame_rate > 0.0 && d->target_frame_rate_ >= frame_rate )
    {
      send_frame = true;
    }
    else if( ts.has_valid_frame() || ts.has_valid_time() )
    {
      send_frame = !d->skip_frame( ts, frame_rate );
    }
  }

  d->is_first_ = false;

  if( d->start_time_ >= 0.0 &&
      ( ts.get_time_seconds() < d->start_time_ ||
        ( d->duration_ > 0.0 &&
          ts.get_time_seconds() > d->start_time_ + d->duration_ ) ) )
  {
    send_frame = false;
  }

  if( d->only_frames_with_dets_ )
  {
    for( size_t i = 0; i < 5; i++ )
    {
      if( has_input_port_edge( d->port_inputs[i] ) )
      {
        try
        {
          if( peek_at_datum_on_port( d->port_inputs[i] )->get_datum<
            kwiver::vital::detected_object_set_sptr >()->empty() )
          {
            send_frame = false;
            break;
          }
        }
        catch( ... )
        {
          continue;
        }
      }
    }
  }

  if( send_frame )
  {
    d->ds_send_counter_++;

    bool in_range = true;

    if( d->frame_range_is_native_ )
    {
      if( orig_ts.has_valid_frame() )
      {
        if( d->start_frame_ >= 0 &&
            orig_ts.get_frame() < d->start_frame_ )
          in_range = false;
        if( d->end_frame_ >= 0 &&
            orig_ts.get_frame() > d->end_frame_ )
          in_range = false;
      }
    }
    else
    {
      int64_t ds_idx = d->ds_send_counter_ - 1;
      if( d->start_frame_ >= 0 && ds_idx < d->start_frame_ )
        in_range = false;
      if( d->end_frame_ >= 0 && ds_idx > d->end_frame_ )
        in_range = false;
    }

    if( !in_range )
      send_frame = false;
  }

  if( send_frame )
  {
    if( d->adjust_timestamps_ )
    {
      if( !d->first_output_seen_ )
      {
        d->first_output_time_ = ts.has_valid_time() ?
          ts.get_time_seconds() : 0.0;
        d->first_output_seen_ = true;
      }
      if( ts.has_valid_time() )
      {
        ts.set_time_seconds(
          ts.get_time_seconds() - d->first_output_time_ );
      }
      ts.set_frame( d->output_counter_++ );
      d->frame_id_map_[ orig_ts.get_frame() ] = ts.get_frame();
    }
    else if( d->renumber_frames_ )
    {
      ts.set_frame( d->output_counter_++ );
      d->frame_id_map_[ orig_ts.get_frame() ] = ts.get_frame();
    }

    if( ts.has_valid_frame() )
    {
      LOG_DEBUG( logger(), "Sending frame " << ts.get_frame() );
    }

    push_to_port_using_trait( timestamp, ts );
    push_to_port_using_trait( original_timestamp, orig_ts );
  }

  for( size_t i = 0; i < 5; i++ )
  {
    if( has_input_port_edge( d->port_inputs[i] ) )
    {
      sprokit::datum_t datum = grab_datum_from_port( d->port_inputs[i] );

      if( datum->type() == sprokit::datum::complete )
      {
        is_finished = true;
      }
      else if( send_frame )
      {
        if( d->renumber_frames_ || d->adjust_timestamps_ )
        {
          datum = d->adjust_track_ids( datum );
        }

        push_datum_to_port( d->port_outputs[i], datum );
      }
    }
  }

  if( is_finished )
  {
    const sprokit::datum_t dat = sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( timestamp, dat );
    push_datum_to_port_using_trait( original_timestamp, dat );

    for( size_t i = 0; i < 5; i++ )
    {
      if( has_input_port_edge( d->port_inputs[i] ) )
      {
        push_datum_to_port( d->port_outputs[i], dat );
      }
    }

    mark_process_as_complete();
  }
}

void downsample_process
::make_ports()
{
  sprokit::process::port_flags_t optional;

  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( frame_rate, optional );

  for( size_t i = 0; i < 5; i++ )
  {
    declare_input_port( priv::port_inputs[i],
                        type_any,
                        optional,
                        port_description_t( "Input data." ) );
  }

  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( original_timestamp, optional );
  declare_output_port_using_trait( frame_rate, optional );

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
  declare_config_using_trait( only_frames_with_dets );
  declare_config_using_trait( start_time );
  declare_config_using_trait( duration );
  declare_config_using_trait( start_frame );
  declare_config_using_trait( end_frame );
  declare_config_using_trait( frame_range_is_native );
  declare_config_using_trait( adjust_timestamps );
}

int downsample_process::priv
::target_frame_count( double time_seconds )
{
  return static_cast< int >( std::floor( time_seconds * target_frame_rate_ + 1e-10 ) );
}

sprokit::datum_t downsample_process::priv
::adjust_track_ids( const sprokit::datum_t& input )
{
  try
  {
    vital::object_track_set_sptr input_set =
      input->get_datum< vital::object_track_set_sptr >();

    if( !input_set || this->frame_id_map_.empty() )
    {
      return input;
    }

    vital::object_track_set_sptr adj_set =
      std::dynamic_pointer_cast< vital::object_track_set >( input_set->clone() );

    if( !adj_set )
    {
      return input;
    }

    for( auto trk : adj_set->tracks() )
    {
      for( auto trk_state : *trk )
      {
        if( !trk_state )
        {
          continue;
        }

        auto iter = frame_id_map_.find( trk_state->frame() );

        if( iter == frame_id_map_.end() )
        {
          trk->remove( trk_state );
        }
        else
        {
          trk_state->set_frame( iter->second );
        }
      }
    }

    return sprokit::datum::new_datum( adj_set );
  }
  catch( ... )
  {
    return input;
  }

  return input;
}

bool downsample_process::priv
::skip_frame( vital::timestamp const& ts,
              double frame_rate )
{
  ds_frame_time_ = ts.has_valid_time() ?
    ts.get_time_seconds() : ds_frame_time_ + 1 / frame_rate;

  if( is_first_ )
  {
    // Triggers always sending the first frame
    last_sent_frame_time_ = ( target_frame_count( ds_frame_time_ ) - 0.5 )
                                        / target_frame_rate_;

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
  , start_time_( -1.0 )
  , duration_( -1.0 )
  , start_frame_( -1 )
  , end_frame_( -1 )
  , frame_range_is_native_( false )
  , adjust_timestamps_( false )
  , ds_send_counter_( 0 )
  , first_output_time_( 0.0 )
  , first_output_seen_( false )
{
}

downsample_process::priv
::~priv()
{
}

}
