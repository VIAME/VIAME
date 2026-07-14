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

  priv( enhance_images& )
   : m_awb_settings()
  {
  }

  ~priv()
  {
  }

  // White balancing runtime state
  std::unique_ptr< auto_white_balancer_base > m_balancer;
  auto_white_balancer_settings m_awb_settings;
};

// --------------------------------------------------------------------------------------
void
enhance_images
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.vxl.enhance_images" );
}

enhance_images
::~enhance_images()
{
}

void
enhance_images
::set_configuration_internal( kv::config_block_sptr )
{
  d->m_awb_settings.white_traverse_factor = get_white_scale_factor();
  d->m_awb_settings.black_traverse_factor = get_black_scale_factor();
  d->m_awb_settings.exp_averaging_factor = get_exp_history_factor();
  d->m_awb_settings.correction_matrix_res = get_matrix_resolution();

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

  if( get_disabled() )
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
      if( get_inversion_enabled() )                                             \
      {                                                                         \
        invert_image( output );                                                 \
      }                                                                         \
                                                                                \
      /* Apply smoothing if enabled */                                          \
      if( get_smoothing_enabled() )                                             \
      {                                                                         \
        vil_gauss_filter_2d( output, output, get_smoothing_std_dev(), get_smoothing_half_width() ); \
      }                                                                         \
                                                                                \
      /* Apply auto white balancing if enabled and 3-channel */                 \
      if( get_auto_white_balance() && output.nplanes() == 3 )                   \
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
      if( get_normalize_brightness() )                                          \
      {                                                                         \
        static mean_illumination_normalization< pix_t > normalizer(             \
          get_brightness_history_length(),                                      \
          get_sampling_rate(),                                                  \
          get_min_percent_brightness(),                                         \
          get_max_percent_brightness() );                                       \
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
