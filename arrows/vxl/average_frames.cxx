// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "average_frames.h"

#include <arrows/vxl/image_container.h>

#include <vital/util/enum_converter.h>

#include <vil/vil_convert.h>
#include <vil/vil_image_view.h>
#include <vil/vil_math.h>
#include <vil/vil_pixel_format.h>

#include <deque>
#include <exception>
#include <memory>

namespace kwiver {

namespace arrows {

namespace vxl {

namespace {

enum averager_mode
{
  AVERAGER_cumulative,
  AVERAGER_window,
  AVERAGER_exponential,
};

ENUM_CONVERTER( averager_converter, averager_mode,
  { "cumulative", AVERAGER_cumulative },
  { "window", AVERAGER_window },
  { "exponential", AVERAGER_exponential } );

/// Base class for all online frame averagers instances
template < typename PixType >
class online_frame_averager
{
public:
  online_frame_averager() : should_round_( false ) {}
  virtual ~online_frame_averager() {}

  /// Process a new frame, returning the current frame average.
  virtual void process_frame( const vil_image_view< PixType >& input,
                              vil_image_view< PixType >& average ) = 0;

  /// Process a new frame, and additionally compute a per-pixel instantaneous
  /// variance estimation, which can be further averaged to estimate the
  /// per-pixel variance over x frames.
  void process_frame( const vil_image_view< PixType >& input,
                      vil_image_view< PixType >& average,
                      vil_image_view< double >& variance );

  /// Reset the internal average.
  virtual void reset() = 0;

protected:
  /// Should we spend a little bit of extra time rounding outputs?
  bool should_round_;

  /// The last average in double form
  vil_image_view< double > last_average_;

  /// Is the resolution of the input image different from prior inputs?
  bool has_resolution_changed( const vil_image_view< PixType >& input );

private:
  /// Temporary buffers used for variance calculations if they're enabled
  vil_image_view< double > dev1_tmp_space_;
  vil_image_view< double > dev2_tmp_space_;
};

/// A cumulative frame averager
template < typename PixType >
class cumulative_frame_averager : public online_frame_averager< PixType >
{
public:
  cumulative_frame_averager( const bool should_round = false );

  virtual ~cumulative_frame_averager() {}

  /// Process a new frame, returning the current frame average.
  virtual void process_frame( const vil_image_view< PixType >& input,
                              vil_image_view< PixType >& average );

  /// Reset the internal average.
  virtual void reset();

protected:
  /// The number of observed frames since the last reset
  unsigned frame_count_;
};

/// An exponential frame averager
template < typename PixType >
class exponential_frame_averager : public online_frame_averager< PixType >
{
public:
  exponential_frame_averager( const bool should_round = false,
                              const double new_frame_weight = 0.5 );

  virtual ~exponential_frame_averager() {}

  /// Process a new frame, returning the current frame average.
  virtual void process_frame( const vil_image_view< PixType >& input,
                              vil_image_view< PixType >& average );

  /// Reset the internal average.
  virtual void reset();

protected:
  /// The exponential averaging coefficient
  double new_frame_weight_;

  /// The number of observed frames since the last reset
  unsigned frame_count_;
};

/// A windowed frame averager
template < typename PixType >
class windowed_frame_averager : public online_frame_averager< PixType >
{
public:
  typedef vil_image_view< PixType > input_type;

  windowed_frame_averager( const bool should_round = false,
                           const unsigned window_length = 20 );

  virtual ~windowed_frame_averager() {}

  /// Process a new frame, returning the current frame average.
  virtual void process_frame( const vil_image_view< PixType >& input,
                              vil_image_view< PixType >& average );

  /// Reset the internal average.
  virtual void reset();

  /// Get number of frames used in the current window
  virtual unsigned frame_count() const;

protected:
  /// Buffer containing pointers to last window_length frames
  std::deque< vil_image_view< PixType > > window_buffer_;
  size_t window_buffer_capacity_;
};

// Shared functionality - process a frame while computing variance
template < typename PixType >
void
online_frame_averager< PixType >
::process_frame( const vil_image_view< PixType >& input,
                 vil_image_view< PixType >& average,
                 vil_image_view< double >& variance )
{
  // Check if this is the first time we have processed a frame of this size
  if( dev1_tmp_space_.ni() != input.ni() ||
      dev1_tmp_space_.nj() != input.nj() ||
      dev1_tmp_space_.nplanes() != input.nplanes() )
  {
    dev1_tmp_space_.set_size( input.ni(), input.nj(), input.nplanes() );
    dev2_tmp_space_.set_size( input.ni(), input.nj(), input.nplanes() );
    variance.set_size( input.ni(), input.nj(), input.nplanes() );
    variance.fill( 0.0 );
    this->process_frame( input, average );
    return;
  }

  // Calculate difference from last average
  vil_math_image_abs_difference( input, this->last_average_, dev1_tmp_space_ );

  // Update internal average
  this->process_frame( input, average );

  // Update the variance
  vil_math_image_abs_difference( input, average, dev2_tmp_space_ );
  vil_math_image_product( dev1_tmp_space_, dev2_tmp_space_, dev1_tmp_space_ );
  variance.deep_copy( dev1_tmp_space_ );
}

template < typename PixType >
bool
online_frame_averager< PixType >
::has_resolution_changed( const vil_image_view< PixType >& input )
{
  return ( input.ni() != this->last_average_.ni() ||
           input.nj() != this->last_average_.nj() ||
           input.nplanes() != this->last_average_.nplanes() );
}

// Helper function to allocate a completely new image, and cast the input image
// to whatever specified type the output image is, scaling by some factor if
// set and rounding if enabled, in one pass.
template < typename inT, typename outT >
void
copy_cast_and_scale( const vil_image_view< inT >& input,
                     vil_image_view< outT >& output,
                     bool rounding_enabled,
                     const inT scale_factor )
{
  // Just deep copy if pixel formats equivalent and there is no scale factor
  if( vil_pixel_format_of( inT() ) == vil_pixel_format_of( outT() ) &&
      scale_factor == 1.0 )
  {
    output.deep_copy( input );
    return;
  }

  // Determine if any rounding would even be beneficial based on source types
  if( rounding_enabled )
  {
    rounding_enabled = !std::numeric_limits< inT >::is_integer &&
                       std::numeric_limits< outT >::is_integer;
  }

  // Resize, cast and copy
  unsigned ni = input.ni(), nj = input.nj(), np = input.nplanes();
  output.set_size( ni, nj, np );

  if( scale_factor == 1.0 && !rounding_enabled )
  {
    vil_convert_cast( input, output );

    // Acording ot the documentation of vil_convert_cast, if the types are the
    // same, the output of vil_convert_cast may be a shallow copy
    if( vil_pixel_format_of( inT() ) == vil_pixel_format_of( outT() ) )
    {
      output.deep_copy( output );
    }
  }
  else
  {
    vil_image_view< inT > scaled;
    scaled.deep_copy( input );
    vil_math_scale_values( scaled, scale_factor );

    if( rounding_enabled )
    {
      vil_convert_round( scaled, output );
    }
    else
    {
      vil_convert_cast( scaled, output );
    }
  }
}

// Cumulative averager
template < typename PixType >
cumulative_frame_averager< PixType >
::cumulative_frame_averager( const bool should_round )
{
  this->should_round_ = should_round;
  this->reset();
}

template < typename PixType >
void
cumulative_frame_averager< PixType >
::process_frame( const vil_image_view< PixType >& input,
                 vil_image_view< PixType >& average )
{
  if( this->has_resolution_changed( input ) )
  {
    this->reset();
  }

  // If this is the first frame observed or there was an indicated reset
  if( this->frame_count_ == 0 )
  {
    vil_convert_cast( input, this->last_average_ );
  }
  // Standard update case
  else
  {
    // Calculate new average - TODO: Non-exponential cumulative average can be
    // modified to be more efficient and prevent precision losses by not using
    // math_add_fraction function. Can also be optimized in the byte case to
    // use integer instead of double operations, but it's good enough for now.
    double scale_factor = 1.0 / ( this->frame_count_ + 1 );

    vil_math_add_image_fraction( this->last_average_,
                                 1.0 - scale_factor,
                                 input,
                                 scale_factor );
  }

  // Copy a completely new image
  copy_cast_and_scale( this->last_average_,
                       average,
                       this->should_round_,
                       1.0 );

  // Increase observed frame count
  this->frame_count_++;
}

template < typename PixType >
void
cumulative_frame_averager< PixType >
::reset()
{
  this->frame_count_ = 0;
}

// Exponential averager
template < typename PixType >
exponential_frame_averager< PixType >
::exponential_frame_averager( const bool should_round,
                              const double new_frame_weight )
{
  this->should_round_ = should_round;
  this->new_frame_weight_ = new_frame_weight;
  this->reset();
}

template < typename PixType >
void
exponential_frame_averager< PixType >
::process_frame( const vil_image_view< PixType >& input,
                 vil_image_view< PixType >& average )
{
  if( this->has_resolution_changed( input ) )
  {
    this->reset();
  }

  // If this is the first frame observed or there was an indicated reset
  if( this->frame_count_ == 0 )
  {
    vil_convert_cast( input, this->last_average_ );
  }
  // Standard update case
  else
  {
    vil_math_add_image_fraction( this->last_average_,
                                 1.0 - new_frame_weight_,
                                 input,
                                 new_frame_weight_ );
  }

  // Copy a completely new image in case we are running in async mode
  copy_cast_and_scale( this->last_average_,
                       average,
                       this->should_round_,
                       1.0 );

  // Increase observed frame count
  this->frame_count_++;
}

template < typename PixType >
void
exponential_frame_averager< PixType >
::reset()
{
  this->frame_count_ = 0;
}

// Windowed averager
template < typename PixType >
windowed_frame_averager< PixType >
::windowed_frame_averager( const bool should_round,
                           const unsigned window_length )
{
  this->window_buffer_capacity_ = window_length;
  this->should_round_ = should_round;
  this->reset();
}

template < typename PixType >
void
windowed_frame_averager< PixType >
::process_frame( const vil_image_view< PixType >& input,
                 vil_image_view< PixType >& average )
{
  if( this->has_resolution_changed( input ) )
  {
    this->reset();
  }

  // Early exit cases: the buffer is currently filling
  const unsigned window_buffer_size = window_buffer_.size();
  if( window_buffer_size == 0 )
  {
    vil_convert_cast( input, this->last_average_ );
  }
  else if( window_buffer_size < window_buffer_capacity_ )
  {
    double src_weight = 1.0 / ( window_buffer_size + 1.0 );
    vil_math_add_image_fraction( this->last_average_, 1.0 - src_weight, input,
                                 src_weight );
  }
  // Standard case, buffer is full
  else
  {
    // Scan image subtracting the last frame, and adding the new one from
    // the previous average
    const unsigned ni = input.ni();
    const unsigned nj = input.nj();
    const unsigned np = input.nplanes();

    // Image A = Removed Entry, B = Added Entry, C = The Average Calculation
    const double scale = 1.0 / window_buffer_size;

    input_type const& tmpA = window_buffer_[ window_buffer_size - 1 ];
    const input_type* imA = &tmpA;
    const input_type* imB = &input;
    vil_image_view< double >* imC = &( this->last_average_ );

    std::ptrdiff_t istepA = imA->istep(), jstepA = imA->jstep(),
                   pstepA = imA->planestep();
    std::ptrdiff_t istepB = imB->istep(), jstepB = imB->jstep(),
                   pstepB = imB->planestep();
    std::ptrdiff_t istepC = imC->istep(), jstepC = imC->jstep(),
                   pstepC = imC->planestep();

    const PixType* planeA = imA->top_left_ptr();
    const PixType* planeB = imB->top_left_ptr();
    double*        planeC = imC->top_left_ptr();

    for( unsigned p = 0; p < np;

         ++p, planeA += pstepA, planeB += pstepB, planeC += pstepC )
    {
      const PixType* rowA = planeA;
      const PixType* rowB = planeB;
      double*        rowC = planeC;

      for( unsigned j = 0; j < nj;

           ++j, rowA += jstepA, rowB += jstepB, rowC += jstepC )
      {
        const PixType* pixelA = rowA;
        const PixType* pixelB = rowB;
        double*        pixelC = rowC;

        for( unsigned i = 0; i < ni;
             ++i, pixelA += istepA, pixelB += istepB, pixelC += istepC )
        {
          *pixelC += scale *
                     ( static_cast< double >( *pixelB ) -
                       static_cast< double >( *pixelA ) );
        }
      }
    }
  }

  // Add to buffer
  this->window_buffer_.push_back( input );
  if( this->window_buffer_.size() > this->window_buffer_capacity_ )
  {
    this->window_buffer_.pop_front();
  }

  // Copy into output
  copy_cast_and_scale( this->last_average_,
                       average,
                       this->should_round_,
                       1.0 );
}

template < typename PixType >
void
windowed_frame_averager< PixType >
::reset()
{
  this->window_buffer_.clear();
}

template < typename PixType >
unsigned
windowed_frame_averager< PixType >
::frame_count() const
{
  return this->window_buffer_.size();
}

} // end anonoymous namespace

// --------------------------------------------------------------------------------------
/// Private implementation class
class average_frames::priv
{
public:
  typedef std::unique_ptr< online_frame_averager< vxl_byte > >
    frame_averager_byte_sptr;
  typedef std::unique_ptr< online_frame_averager< double > >
    frame_averager_float_sptr;

  priv()
    : type( AVERAGER_window )
      , window_size( 10 )
      , exp_weight( 0.3 )
      , round( false )
      , output_variance( false )
      , variance_scale( 0.0 )
  {
  }

  ~priv()
  {
  }

  // Internal parameters/settings
  averager_mode type;
  unsigned window_size;
  double exp_weight;
  bool round;
  bool output_variance;
  double variance_scale;

  // The actual frame averager
  frame_averager_byte_sptr byte_averager;
  frame_averager_float_sptr float_averager;

  // Load model, special optimizations are in place for the byte case
  void
  load_model( bool is_byte = true )
  {
    if( ( is_byte && byte_averager ) || ( !is_byte && float_averager ) )
    {
      return;
    }

    switch( type )
    {
      case AVERAGER_window:
      {
        if( is_byte )
        {
          byte_averager.reset(
              new windowed_frame_averager< vxl_byte >( round, window_size ) );
        }
        else
        {
          float_averager.reset(
              new windowed_frame_averager< double >( round, window_size ) );
        }
        break;
      }
      case AVERAGER_cumulative:
      {
        if( is_byte )
        {
          byte_averager.reset(
              new cumulative_frame_averager< vxl_byte >( round ) );
        }
        else
        {
          float_averager.reset(
              new cumulative_frame_averager< double >( round ) );
        }
        break;
      }
      case AVERAGER_exponential:
      {
        if( exp_weight <= 0 || exp_weight >= 1 )
        {
          throw std::runtime_error(
                  "Invalid exponential averaging coefficient!" );
        }

        if( is_byte )
        {
          byte_averager.reset(
              new exponential_frame_averager< vxl_byte >( round,
                                                          exp_weight ) );
        }
        else
        {
          float_averager.reset(
              new exponential_frame_averager< double >( round, exp_weight ) );
        }
        break;
      }
      default:
      {
        throw std::runtime_error( "Invalid averaging type!" );
      }
    }
  }

  // Compute the updated average with the current frame
  // return the average or the variance
  kwiver::vital::image_container_sptr
  process_frame( vil_image_view< double > input )
  {
    load_model( false );

    if( !output_variance )
    {
      vil_image_view< double > output;
      float_averager->process_frame( input, output );
      return std::make_shared< vxl::image_container >( output );
    }
    else
    {
      vil_image_view< double > tmp, output;
      float_averager->process_frame( input, tmp, output );
      return std::make_shared< vxl::image_container >( output );
    }
  }
};

// ----------------------------------------------------------------------------
average_frames
::average_frames()
  : d( new priv() )
{
  attach_logger( "arrows.vxl.average_frames" );
}

// ----------------------------------------------------------------------------
average_frames
::~average_frames()
{
}

// ----------------------------------------------------------------------------
vital::config_block_sptr
average_frames
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "type", averager_converter().to_string( d->type ),
                     "Operating mode of this filter, possible values: " +
                     averager_converter().element_name_string() );
  config->set_value( "window_size", d->window_size,
                     "The window size if computing a windowed moving average." );
  config->set_value( "exp_weight", d->exp_weight,
                     "Exponential averaging coefficient if computing an exp average." );
  config->set_value( "round", d->round,
                     "Should we spend a little extra time rounding when possible?" );
  config->set_value( "output_variance", d->output_variance,
                     "If set, will compute an estimated variance for each pixel which "
                     "will be outputted as either a double-precision or byte image." );

  return config;
}

// ----------------------------------------------------------------------------
void
average_frames
::set_configuration( vital::config_block_sptr in_config )
{
  // Starting with our generated vital::config_block to ensure that assumed
  // values are present. An alternative is to check for key presence before
  // performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  // Settings for averaging
  d->type = config->get_enum_value< averager_converter >( "type" );
  d->window_size = config->get_value< unsigned >( "window_size" );
  d->exp_weight = config->get_value< double >( "exp_weight" );
  d->round = config->get_value< bool >( "round" );
  d->output_variance = config->get_value< bool >( "output_variance" );
}

// ----------------------------------------------------------------------------
bool
average_frames
::check_configuration( vital::config_block_sptr config ) const
{
  auto const& type = config->get_enum_value< averager_converter >( "type" );
  if( !( type == AVERAGER_cumulative ||
         type == AVERAGER_window ||
         type == AVERAGER_exponential ) )
  {
    return false;
  }
  else if( type == AVERAGER_exponential )
  {
    double exp_weight = config->get_value< double >( "exp_weight" );
    if( exp_weight <= 0 || exp_weight > 1 )
    {
      return false;
    }
  }
  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
average_frames
::filter( kwiver::vital::image_container_sptr image_data )
{
  // Perform Basic Validation
  if( !image_data )
  {
    return image_data;
  }

  // Get input image
  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl( image_data->get_image() );

  // Perform different actions based on input type
#define HANDLE_CASE( T )                                               \
  case T:                                                              \
  {                                                                    \
    typedef vil_pixel_format_type_of< T >::component_type pix_t;       \
    vil_image_view< pix_t > uncast_input = view;                       \
    vil_image_view< double > input;                                    \
    vil_convert_cast( uncast_input, input );                           \
                                                                       \
    d->process_frame( input );                                         \
    break;                                                             \
  }                                                                    \

  switch( view->pixel_format() )
  {
    HANDLE_CASE( VIL_PIXEL_FORMAT_BOOL );
    HANDLE_CASE( VIL_PIXEL_FORMAT_SBYTE );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_16 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_INT_16 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_32 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_INT_32 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_UINT_64 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_INT_64 );
    HANDLE_CASE( VIL_PIXEL_FORMAT_FLOAT );
    HANDLE_CASE( VIL_PIXEL_FORMAT_DOUBLE );
#undef HANDLE_CASE

    case VIL_PIXEL_FORMAT_BYTE:
    {
      // Default byte case
      vil_image_view< vxl_byte > input = view;

      d->load_model( true );

      if( !d->output_variance )
      {
        vil_image_view< vxl_byte > output;
        d->byte_averager->process_frame( input, output );
        return std::make_shared< vxl::image_container >( output );
      }
      else
      {
        vil_image_view< vxl_byte > tmp;
        vil_image_view< double > output;
        d->byte_averager->process_frame( input, tmp, output );
        return std::make_shared< vxl::image_container >( output );
      }
      break;
    }

    default:
      // The image type was not one we handle
      LOG_ERROR( logger(), "Invalid input format "      << view->pixel_format()
                                                        << " type received" );
      return kwiver::vital::image_container_sptr();
  }

  // Code not reached, prevent warning
  return kwiver::vital::image_container_sptr();
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
