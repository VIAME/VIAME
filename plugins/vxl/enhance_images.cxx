/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "enhance_images.h"
#include "perform_white_balancing.h"

#include <arrows/vxl/image_container.h>

#include <vil/vil_pixel_format.h>
#include <vil/vil_image_view.h>
#include <vil/vil_copy.h>
#include <vil/vil_math.h>
#include <vil/vil_rgb.h>
#include <vil/vil_transform.h>
#include <vil/algo/vil_gauss_filter.h>

#include <exception>
#include <deque>

namespace viame {

namespace kv = kwiver::vital;
namespace vxl = kwiver::arrows::vxl;

namespace {

// Simple scaling functor for increasing/decreasing image brightness
template <class PixType>
class illum_scale_functor
{
  double s_;

public:

  illum_scale_functor( double s ) : s_( s ) {}

  PixType operator()( PixType x ) const
  {
    double temp_ = 0.5 + s_ * x;
    if( temp_ > std::numeric_limits<PixType>::max() )
    {
      return std::numeric_limits<PixType>::max();
    }
    return static_cast<PixType>( temp_ );
  }
};

// Specialization for float
template <>
class illum_scale_functor<float>
{
  double s_;

public:

  illum_scale_functor( double s ) : s_( s ) {}

  float operator()( float x ) const
  {
    return std::min( 1.0f, static_cast<float>( s_ * x ) );
  }
};

// Specialization for double
template <>
class illum_scale_functor<double>
{
  double s_;

public:

  illum_scale_functor( double s ) : s_( s ) {}

  double operator()( double x ) const
  {
    return std::min( 1.0, s_ * x );
  }
};


// Abstract base class for illumination normalizers
template < class PixType >
class illumination_normalization
{
public:
  illumination_normalization() {}
  virtual ~illumination_normalization() {}

  virtual vil_image_view<PixType> operator()( vil_image_view<PixType> const& img,
                                              bool deep_copy = false ) = 0;
  virtual void reset() = 0;
};


// Normalizes illumination based on mean of pixel values
template < class PixType >
class mean_illumination_normalization : public illumination_normalization< PixType >
{
public:

  mean_illumination_normalization( unsigned window_length = 20,
                                   unsigned sampling_rate = 4,
                                   double min_illum_allowed = 0.0,
                                   double max_illum_allowed = 1.0 )
  : window_length_( window_length ),
    sampling_rate_( sampling_rate ),
    min_illum_allowed_( min_illum_allowed ),
    max_illum_allowed_( max_illum_allowed )
  {
    this->reset();
  }

  vil_image_view<PixType> operator()( vil_image_view<PixType> const& img, bool deep_copy = false );

  void reset();

  double calculate_mean( vil_image_view<PixType> const& img ) const;

protected:

  unsigned window_length_;
  unsigned sampling_rate_;
  double min_illum_allowed_;
  double max_illum_allowed_;

  std::deque< double > illum_history_;
};


template<class PixType>
double
mean_illumination_normalization<PixType>
::calculate_mean( vil_image_view<PixType> const & img ) const
{
  // Downsample the image
  vil_image_view<PixType> downsampled( img.top_left_ptr(),
                                       1+(img.ni()-1)/sampling_rate_,
                                       1+(img.nj()-1)/sampling_rate_,
                                       img.nplanes(),
                                       sampling_rate_ * img.istep(),
                                       sampling_rate_ * img.jstep(),
                                       img.planestep() );

  if( downsampled.size() == 0 )
  {
    return 0.0;
  }

  const double pixels_per_plane = static_cast<double>(downsampled.ni()*downsampled.nj());

  std::vector< double > chan_avg( downsampled.nplanes(), 0.0 );

  for( unsigned p=0;p<img.nplanes();++p )
  {
    vil_math_sum( chan_avg[p], downsampled, p );
  }

  // Special case for RGB, use standard intensity
  if( img.nplanes() == 3 )
  {
    return vil_rgb<double>(chan_avg[0], chan_avg[1], chan_avg[2]).grey() / pixels_per_plane;
  }

  // For all other channel amounts average each channel
  for( unsigned p=1; p<img.nplanes(); ++p )
  {
    chan_avg[0] += chan_avg[p];
  }
  return chan_avg[0] / ( pixels_per_plane*img.nplanes() );
}

template<class PixType>
vil_image_view<PixType>
mean_illumination_normalization<PixType>
::operator()( vil_image_view<PixType> const & img, bool deep_copy )
{
  vil_image_view<PixType> result;
  if( deep_copy )
  {
    result.deep_copy(img);
  }
  else
  {
    result = img;
  }

  // Calculate current illumination
  double est = this->calculate_mean( img );
  illum_history_.push_back( est );
  if( illum_history_.size() > window_length_ )
  {
    illum_history_.pop_front();
  }

  // Calculate desired average illumination
  double avg = 0.0;
  for( unsigned i = 0; i < illum_history_.size(); i++ )
  {
    avg += illum_history_[i];
  }
  avg = avg / illum_history_.size();

  // Threshold average illumination based on type
  double min_threshold = min_illum_allowed_ * default_white_point<PixType>::value();
  double max_threshold = max_illum_allowed_ * default_white_point<PixType>::value();

  if( avg < min_threshold )
  {
    avg = min_threshold;
  }
  if( avg > max_threshold )
  {
    avg = max_threshold;
  }

  // If empty frames received, simply reset and exit
  if( avg == 0.0 || est == 0.0 )
  {
    this->reset();
    return result;
  }

  // Adjust image brightness
  vil_transform( result, illum_scale_functor<PixType>(avg/est) );
  return result;
}

template<class PixType>
void
mean_illumination_normalization<PixType>
::reset()
{
  illum_history_.clear();
}


} // end anonoymous namespace


// --------------------------------------------------------------------------------------
/// Private implementation class
class enhance_images::priv
{
public:

  priv()
   : m_disabled( false ),
     m_smoothing_enabled( false ),
     m_std_dev( 0.6 ),
     m_half_width( 2 ),
     m_inversion_enabled( false ),
     m_awb_enabled( true ),
     m_normalize_brightness( true ),
     m_sampling_rate( 2 ),
     m_brightness_history_length( 10 ),
     m_min_percent_brightness( 0.10 ),
     m_max_percent_brightness( 0.90 )
  {
  }

  ~priv()
  {
  }

  // Settings
  bool m_disabled;

  // Smoothing
  bool m_smoothing_enabled;
  double m_std_dev;
  unsigned m_half_width;

  // Inversion
  bool m_inversion_enabled;

  // White balancing
  bool m_awb_enabled;
  std::unique_ptr< auto_white_balancer_base > m_balancer;
  auto_white_balancer_settings m_awb_settings;

  // Illumination normalization
  bool m_normalize_brightness;
  unsigned m_sampling_rate;
  unsigned m_brightness_history_length;
  double m_min_percent_brightness;
  double m_max_percent_brightness;
};

// --------------------------------------------------------------------------------------
enhance_images
::enhance_images()
: d( new priv() )
{
  attach_logger( "viame.vxl.enhance_images" );
}

enhance_images
::~enhance_images()
{
}

kv::config_block_sptr
enhance_images
::get_configuration() const
{
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  config->set_value( "disabled", d->m_disabled,
    "Completely disable this process and pass the input image" );

  config->set_value( "smoothing_enabled", d->m_smoothing_enabled,
    "Perform extra internal smoothing on the input" );
  config->set_value( "smoothing_std_dev", d->m_std_dev,
    "Std dev for internal gaussian smoothing" );
  config->set_value( "smoothing_half_width", d->m_half_width,
    "Half width for internal gaussian smoothing" );

  config->set_value( "inversion_enabled", d->m_inversion_enabled,
    "Should we invert the input image?" );

  config->set_value( "auto_white_balance", d->m_awb_enabled,
    "Whether or not auto-white balancing is enabled" );
  config->set_value( "white_scale_factor", d->m_awb_settings.white_traverse_factor,
    "A measure of how much to over or under correct white reference points." );
  config->set_value( "black_scale_factor", d->m_awb_settings.black_traverse_factor,
    "A measure of how much to over or under correct black reference points." );
  config->set_value( "exp_history_factor", d->m_awb_settings.exp_averaging_factor,
    "The exponential averaging factor for correction matrices" );
  config->set_value( "matrix_resolution", d->m_awb_settings.correction_matrix_res,
    "The resolution of the correction matrix" );

  config->set_value( "normalize_brightness", d->m_normalize_brightness,
    "If enabled, will attempt to stabilize video illumination" );
  config->set_value( "sampling_rate", d->m_sampling_rate,
    "The sampling rate used when approximating the mean scene illumination." );
  config->set_value( "brightness_history_length", d->m_brightness_history_length,
    "Attempt to stabilize the brightness using data from the last x frames." );
  config->set_value( "min_percent_brightness", d->m_min_percent_brightness,
    "The minimum allowed average brightness for an image." );
  config->set_value( "max_percent_brightness", d->m_max_percent_brightness,
    "The maximum allowed average brightness for an image." );

  return config;
}

void
enhance_images
::set_configuration( kv::config_block_sptr in_config )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d->m_disabled = config->get_value< bool >( "disabled" );

  d->m_smoothing_enabled = config->get_value< bool >( "smoothing_enabled" );
  d->m_std_dev = config->get_value< double >( "smoothing_std_dev" );
  d->m_half_width = config->get_value< unsigned >( "smoothing_half_width" );

  d->m_inversion_enabled = config->get_value< bool >( "inversion_enabled" );

  d->m_awb_enabled = config->get_value< bool >( "auto_white_balance" );
  d->m_awb_settings.white_traverse_factor =
    config->get_value< double >( "white_scale_factor" );
  d->m_awb_settings.black_traverse_factor =
    config->get_value< double >( "black_scale_factor" );
  d->m_awb_settings.exp_averaging_factor  =
    config->get_value< double >( "exp_history_factor" );
  d->m_awb_settings.correction_matrix_res =
    config->get_value< unsigned> ( "matrix_resolution"  );

  d->m_normalize_brightness = config->get_value< bool >( "normalize_brightness" );
  d->m_sampling_rate = config->get_value< unsigned >( "sampling_rate" );
  d->m_brightness_history_length = config->get_value< unsigned >( "brightness_history_length" );
  d->m_min_percent_brightness = config->get_value< double >( "min_percent_brightness" );
  d->m_max_percent_brightness = config->get_value< double >( "max_percent_brightness" );

  // Validate parameters
  if( d->m_awb_settings.exp_averaging_factor > 1.0 )
  {
    throw std::runtime_error( "Invalid exponential averaging weight" );
  }
  if( d->m_awb_settings.correction_matrix_res > 200 )
  {
    throw std::runtime_error( "Correction matrix resolution is too large!" );
  }
}

bool
enhance_images
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}


// Invert image helper template
template <typename PixType>
void invert_image( vil_image_view<PixType>& img )
{
  PixType max_val = default_white_point<PixType>::value();

  for( unsigned j = 0; j < img.nj(); ++j )
  {
    for( unsigned i = 0; i < img.ni(); ++i )
    {
      for( unsigned p = 0; p < img.nplanes(); ++p )
      {
        img(i, j, p) = max_val - img(i, j, p);
      }
    }
  }
}


// Perform enhancement operation
kv::image_container_sptr
enhance_images
::filter( kv::image_container_sptr image_data )
{
  // Perform Basic Validation
  if( !image_data )
  {
    return image_data;
  }

  if( d->m_disabled )
  {
    return image_data;
  }

  // Get input image
  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl( image_data->get_image() );

  // Perform different actions based on input type
#define HANDLE_CASE(T)                                                          \
  case T:                                                                       \
    {                                                                           \
      typedef vil_pixel_format_type_of<T >::component_type pix_t;               \
      vil_image_view< pix_t > input = view;                                     \
                                                                                \
      vil_image_view< pix_t > output;                                           \
      vil_copy_deep( input, output );                                           \
                                                                                \
      /* Apply inversion if enabled */                                          \
      if( d->m_inversion_enabled )                                              \
      {                                                                         \
        invert_image( output );                                                 \
      }                                                                         \
                                                                                \
      /* Apply smoothing if enabled */                                          \
      if( d->m_smoothing_enabled )                                              \
      {                                                                         \
        vil_gauss_filter_2d( output, output, d->m_std_dev, d->m_half_width );   \
      }                                                                         \
                                                                                \
      /* Apply auto white balancing if enabled and 3-channel */                 \
      if( d->m_awb_enabled && output.nplanes() == 3 )                           \
      {                                                                         \
        auto_white_balancer< pix_t >* balancer =                                \
          dynamic_cast< auto_white_balancer< pix_t >* >( d->m_balancer.get() ); \
                                                                                \
        if( !balancer )                                                         \
        {                                                                       \
          balancer = new auto_white_balancer< pix_t >();                        \
          balancer->configure( d->m_awb_settings );                             \
          d->m_balancer = std::unique_ptr< auto_white_balancer_base >( balancer );\
        }                                                                       \
                                                                                \
        balancer->apply( output );                                              \
      }                                                                         \
                                                                                \
      /* Apply illumination normalization if enabled */                         \
      if( d->m_normalize_brightness )                                           \
      {                                                                         \
        static mean_illumination_normalization< pix_t > normalizer(             \
          d->m_brightness_history_length,                                       \
          d->m_sampling_rate,                                                   \
          d->m_min_percent_brightness,                                          \
          d->m_max_percent_brightness );                                        \
        output = normalizer( output, false );                                   \
      }                                                                         \
                                                                                \
      return std::make_shared< vxl::image_container >( output );                \
    }                                                                           \
    break;                                                                      \

  switch( view->pixel_format() )
  {
    HANDLE_CASE(VIL_PIXEL_FORMAT_BYTE);
    HANDLE_CASE(VIL_PIXEL_FORMAT_UINT_16);
    HANDLE_CASE(VIL_PIXEL_FORMAT_FLOAT);
    HANDLE_CASE(VIL_PIXEL_FORMAT_DOUBLE);
#undef HANDLE_CASE

  default:
    throw std::runtime_error( "Unsupported type received" );
  }

  // Code not reached, prevent warning
  return kv::image_container_sptr();
}

} // end namespace viame
