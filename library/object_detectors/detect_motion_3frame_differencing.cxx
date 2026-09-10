// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of ocv::detect_moiion_3frame_differencing

#include <deque>

#include "detect_motion_3frame_differencing.h"

#include <kwiversys/SystemTools.hxx>
#include <viame/algorithm_framework/exceptions.h>
#include <viame/core_types/matrix.h>
#include <viame/algorithm_framework/vital_config.h>

#include <image_ops/morphology.h>
#include <image_ops/pixel.h>

#include <viame/core_types/image_container.h>
#include <viame/video_io/codecs/image_codec.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace kwiver {

namespace arrows {

namespace ocv {

using namespace kwiver::vital;

namespace io = viame::image_ops;
namespace kv = kwiver::vital;

namespace {

/// A difference image, kept in float because the jittered path needs it.
///
/// The unjittered path does not: `cv::absdiff` on two byte images gives a
/// byte image, and the sum that follows is byte arithmetic that **saturates**
/// at 255 and clamps at 0. That saturation is part of the answer -- a bright
/// enough three-frame difference is clipped rather than wrapped -- so the
/// eight bit path is kept eight bit rather than being widened for tidiness.
using plane_f = kv::image_of< float >;

/// Root mean square over the planes, as `rms_over_channels` computed it.
///
/// Each plane's square is divided by nine before summing -- the comment in
/// the original says "3^2 so that the difference has the same scale as a
/// mono image" -- and the result is rounded into a byte the way
/// `cv::Mat::convertTo` rounds, which is half to even.
template < typename T >
kv::image_of< uint8_t >
rms_over_planes( kv::image_of< T > const& source )
{
  kv::image_of< uint8_t > out( source.width(), source.height(), 1 );

  for( size_t j = 0; j < source.height(); ++j )
  {
    for( size_t i = 0; i < source.width(); ++i )
    {
      double total = 0.0;

      for( size_t p = 0; p < source.depth(); ++p )
      {
        auto const value = static_cast< double >( source( i, j, p ) );
        total += value * value / 3.0;
      }

      out( i, j, 0 ) = io::saturate_pixel_even< uint8_t >(
        std::sqrt( total ) );
    }
  }

  return out;
}

} // namespace

// ----------------------------------------------------------------------------
/// Private implementation class
class detect_motion_3frame_differencing::priv
{
  io::structuring_element m_jitter_struct_el;
  std::deque< kv::image_of< uint8_t > > m_frames;
  int m_debug_counter = 0;
  detect_motion_3frame_differencing& parent;

public:
  /// Parameters
  std::string
  m_debug_dir() const { return parent.get_debug_dir(); }
  std::size_t
  m_frame_separation() const { return parent.get_frame_separation(); }
  int
  m_jitter_radius() const { return parent.get_jitter_radius(); }
  double
  m_max_foreground_fract() const { return parent.get_max_foreground_fract(); }

  double
  m_max_foreground_fract_thresh() const
  {
    return parent.get_max_foreground_fract_thresh();
  }

  kwiver::vital::logger_handle_t m_logger;
  bool m_output_to_debug_dir = false;

  /// Constructor
  priv( detect_motion_3frame_differencing& parent )
    : parent( parent )
  {}

  /// Flush the image queue.
  void
  reset()
  {
    m_frames.clear();
  }

  ///
  /// @brief Calculates a jittered difference between img1 and img2
  ///
  /// For each pixel in img1, the minimum absolute difference ||img1-b|| is
  /// calculated, where b is drawn from a neighborhood (defined by
  /// m_jitter_radius) around the equivalent pixel in img2.
  ///
  /// Following "Detecting and Tracking All Moving Objects in Wide-Area
  /// Aerial Video", equation 2.
  ///
  /// The result is float, and so is the arithmetic that follows it. The
  /// unjittered path is byte instead -- see `difference_bytes`.
  ///
  /// @param img1 first image
  /// @param img2 second image
  kv::image_of< float >
  jittered_difference( kv::image_of< uint8_t > const& img1,
                       kv::image_of< uint8_t > const& img2 )
  {
    if( m_jitter_struct_el.empty() )
    {
      auto const side = 2 * m_jitter_radius() + 1;
      m_jitter_struct_el = io::rect_element( side, side );
    }

    auto const local_max = io::grey_dilate( img2, m_jitter_struct_el );
    auto const local_min = io::grey_erode( img2, m_jitter_struct_el );

    kv::image_of< float > out( img1.width(), img1.height(), img1.depth() );

    for( size_t p = 0; p < img1.depth(); ++p )
    {
      for( size_t j = 0; j < img1.height(); ++j )
      {
        for( size_t i = 0; i < img1.width(); ++i )
        {
          auto const one = static_cast< float >( img1( i, j, p ) );

          // Negatives clipped to zero, which is what THRESH_TOZERO did
          auto const below =
            std::max( 0.0f,
                      static_cast< float >( local_min( i, j, p ) ) - one );
          auto const above =
            std::max( 0.0f,
                      one - static_cast< float >( local_max( i, j, p ) ) );

          out( i, j, p ) = std::max( below, above );
        }
      }
    }

    return out;
  }

  /// The unjittered difference, which is `cv::absdiff` on two byte images.
  static kv::image_of< uint8_t >
  difference_bytes( kv::image_of< uint8_t > const& img1,
                    kv::image_of< uint8_t > const& img2 )
  {
    kv::image_of< uint8_t > out( img1.width(), img1.height(), img1.depth() );

    for( size_t p = 0; p < img1.depth(); ++p )
    {
      for( size_t j = 0; j < img1.height(); ++j )
      {
        for( size_t i = 0; i < img1.width(); ++i )
        {
          auto const a = static_cast< int >( img1( i, j, p ) );
          auto const b = static_cast< int >( img2( i, j, p ) );
          out( i, j, p ) = static_cast< uint8_t >( std::abs( a - b ) );
        }
      }
    }

    return out;
  }

  /// The foreground mask, before the plane reduction.
  ///
  /// `| |A - C| + |C - B| - |A - B| |`, and the type it is computed in
  /// matters: the unjittered path is byte arithmetic that **saturates** at
  /// 255 on the addition and clamps at 0 on the subtraction, exactly as
  /// OpenCV's byte `Mat` arithmetic does. Widening it would change the
  /// answer wherever the three-frame difference is bright.
  kv::image_of< uint8_t >
  combine_bytes( kv::image_of< uint8_t > const& ac,
                 kv::image_of< uint8_t > const& cb,
                 kv::image_of< uint8_t > const& ab )
  {
    kv::image_of< uint8_t > out( ac.width(), ac.height(), ac.depth() );

    for( size_t p = 0; p < ac.depth(); ++p )
    {
      for( size_t j = 0; j < ac.height(); ++j )
      {
        for( size_t i = 0; i < ac.width(); ++i )
        {
          auto const sum = std::min(
            255, static_cast< int >( ac( i, j, p ) ) +
                 static_cast< int >( cb( i, j, p ) ) );

          auto const difference =
            std::max( 0, sum - static_cast< int >( ab( i, j, p ) ) );

          // `cv::abs` on an unsigned result is the identity; kept for the
          // shape of the formula rather than because it can do anything
          out( i, j, p ) = static_cast< uint8_t >( difference );
        }
      }
    }

    return out;
  }

  kv::image_of< float >
  combine_floats( kv::image_of< float > const& ac,
                  kv::image_of< float > const& cb,
                  kv::image_of< float > const& ab )
  {
    kv::image_of< float > out( ac.width(), ac.height(), ac.depth() );

    for( size_t p = 0; p < ac.depth(); ++p )
    {
      for( size_t j = 0; j < ac.height(); ++j )
      {
        for( size_t i = 0; i < ac.width(); ++i )
        {
          out( i, j, p ) = std::abs(
            ac( i, j, p ) + cb( i, j, p ) - ab( i, j, p ) );
        }
      }
    }

    return out;
  }

  /// Write one debug frame, if a debug directory was configured.
  void
  save_debug( kv::image const& image, char const* what )
  {
    auto const name = m_debug_dir() + "/" +
                      std::to_string( m_debug_counter ) + what + ".tif";

    try
    {
      viame::codecs::write( name, image );
    }
    catch( std::exception const& e )
    {
      LOG_WARN( m_logger, "Could not write " << name << ": " << e.what() );
    }
  }

  kv::image_of< uint8_t >
  process_image( kv::image_of< uint8_t > const& source )
  {
    // Images are in temporal order A (oldest), B, C (newest).
    m_frames.push_front( source );

    if( m_frames.size() < 2 * m_frame_separation() )
    {
      LOG_TRACE(
        m_logger, "Haven't collected enough frames yet, so setting "
                  "foreground mask to all zeros." );

      kv::image_of< uint8_t > empty( source.width(), source.height(), 1 );

      for( size_t j = 0; j < empty.height(); ++j )
      {
        for( size_t i = 0; i < empty.width(); ++i )
        {
          empty( i, j, 0 ) = 0;
        }
      }

      return empty;
    }

    LOG_TRACE( m_logger, "Getting frame from end of queue" );
    auto const imgA = m_frames.back();
    LOG_TRACE( m_logger, "Getting frame at index frame_separation" );
    auto const imgB = m_frames[ m_frame_separation() ];
    auto const imgC = m_frames.front();

    if( m_frames.size() > 2 * m_frame_separation() )
    {
      LOG_TRACE( m_logger, "Removing frame from end of queue" );
      m_frames.pop_back();
    }

    kv::image_of< uint8_t > fgmask;

    if( m_jitter_radius() == 0 )
    {
      auto const ac = difference_bytes( imgA, imgC );
      auto const cb = difference_bytes( imgC, imgB );
      auto const ab = difference_bytes( imgA, imgB );

      auto const combined = combine_bytes( ac, cb, ab );

      if( m_output_to_debug_dir )
      {
        save_debug( kv::image( imgA ), "imgA" );
        save_debug( kv::image( imgB ), "imgB" );
        save_debug( kv::image( imgC ), "imgC" );
        save_debug( kv::image( ac ), "AminusC" );
        save_debug( kv::image( cb ), "CminusB" );
        save_debug( kv::image( ab ), "AminusB" );
        save_debug( kv::image( combined ), "fgmask" );
        ++m_debug_counter;
      }

      fgmask = ( combined.depth() > 1 ) ? rms_over_planes( combined )
                                        : combined;
    }
    else
    {
      auto const ac = jittered_difference( imgA, imgC );
      auto const cb = jittered_difference( imgC, imgB );
      auto const ab = jittered_difference( imgA, imgB );

      auto const combined = combine_floats( ac, cb, ab );

      if( m_output_to_debug_dir )
      {
        save_debug( kv::image( imgA ), "imgA" );
        save_debug( kv::image( imgB ), "imgB" );
        save_debug( kv::image( imgC ), "imgC" );
        ++m_debug_counter;
      }

      if( combined.depth() > 1 )
      {
        LOG_TRACE(
          m_logger, "Converting multichannel foreground mask to single "
                    "channel" );
      }

      fgmask = rms_over_planes( combined );
    }

    if( IS_TRACE_ENABLED( m_logger ) )
    {
      int lowest = 255;
      int highest = 0;

      for( size_t j = 0; j < fgmask.height(); ++j )
      {
        for( size_t i = 0; i < fgmask.width(); ++i )
        {
          auto const value = static_cast< int >( fgmask( i, j, 0 ) );
          lowest = std::min( lowest, value );
          highest = std::max( highest, value );
        }
      }

      LOG_TRACE(
        m_logger, "heat map min: " + std::to_string( lowest ) +
        " max: " + std::to_string( highest ) );
    }

    if( m_max_foreground_fract() < 1 )
    {
      auto const total_pixels = fgmask.width() * fgmask.height();
      auto const max_fg_pixels = static_cast< size_t >(
        static_cast< double >( total_pixels ) * m_max_foreground_fract() );

      size_t nonzero_pixels = 0;

      for( size_t j = 0; j < fgmask.height(); ++j )
      {
        for( size_t i = 0; i < fgmask.width(); ++i )
        {
          if( static_cast< double >( fgmask( i, j, 0 ) ) >
              m_max_foreground_fract_thresh() )
          {
            ++nonzero_pixels;
          }
        }
      }

      LOG_TRACE(
        m_logger, ( double ) nonzero_pixels / ( double ) total_pixels * 100 <<
          "% foreground pixels." );

      if( nonzero_pixels > max_fg_pixels )
      {
        LOG_DEBUG(
          m_logger, "Foreground pixels exceed maximum set to " <<
            m_max_foreground_fract() * 100 << "%, something must have "
                                            "failed. Resetting background model." );

        // Reset background model, but wait until next iteration to start
        // updating it because the current frame might be bad.
        reset();

        for( size_t j = 0; j < fgmask.height(); ++j )
        {
          for( size_t i = 0; i < fgmask.width(); ++i )
          {
            fgmask( i, j, 0 ) = 0;
          }
        }
      }
    }

    return fgmask;
  }

  /// Set up debug directory
  void
  setup_debug_dir()
  {
    LOG_DEBUG( m_logger, "Creating debug directory: " + m_debug_dir() );
    kwiversys::SystemTools::MakeDirectory( m_debug_dir() );
    m_output_to_debug_dir = true;
  }
};

void
detect_motion_3frame_differencing
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d_ );
  attach_logger( "arrows.ocv.detect_motion_3frame_differencing" );
  d_->m_logger = logger();
  d_->reset();
}

/// Destructor
detect_motion_3frame_differencing
::~detect_motion_3frame_differencing() noexcept
{}

/// Set this algo's properties via a config block
void
detect_motion_3frame_differencing
::set_configuration_internal( [[maybe_unused]] vital::config_block_sptr config )
{
  if( this->get_frame_separation() < 0 )
  {
    VITAL_THROW(
      algorithm_configuration_exception, interface_name(), impl_name(),
      "frame_separation must be an "
      "integer greater than 0." );
  }

  if( this->get_jitter_radius() < 0 )
  {
    VITAL_THROW(
      algorithm_configuration_exception, interface_name(), impl_name(),
      "m_jitter_radius must be an "
      "integer greater than 0." );
  }

  if( this->get_max_foreground_fract() < 0 ||
      this->get_max_foreground_fract() > 1 )
  {
    VITAL_THROW(
      algorithm_configuration_exception, interface_name(), impl_name(),
      "max_foreground_fract must be in "
      "the range 0-1." );
  }

  if( this->get_max_foreground_fract() != 1 &&
      this->get_max_foreground_fract_thresh() < 0 )
  {
    VITAL_THROW(
      algorithm_configuration_exception, interface_name(), impl_name(),
      "max_foreground_fract_thresh must "
      "be set as a positive value." );
  }

  if( !( this->get_debug_dir().empty() || this->get_debug_dir() == "" ) )
  {
    d_->setup_debug_dir();
  }

  LOG_DEBUG(
    logger(),
    "frame_separation: " << std::to_string( this->get_frame_separation() ) );
  LOG_DEBUG(
    logger(),
    "jitter_radius: " << std::to_string( this->get_jitter_radius() ) );
  LOG_DEBUG(
    logger(),
    "max_foreground_fract: " <<
      std::to_string( this->get_max_foreground_fract() ) );
  LOG_DEBUG(
    logger(),
    "max_foreground_fract_thresh: " <<
      std::to_string( this->get_max_foreground_fract_thresh() ) );
  LOG_DEBUG( logger(), "debug_dir: " << this->get_debug_dir() );
}

bool
detect_motion_3frame_differencing
::check_configuration( [[maybe_unused]] vital::config_block_sptr config ) const
{
  return true;
}

/// Detect motion from a sequence of images
image_container_sptr
detect_motion_3frame_differencing
::process_image(
  [[maybe_unused]] const timestamp& ts,
  const image_container_sptr image,
  bool reset_model )
{
  if( !image )
  {
    VITAL_THROW(
      vital::invalid_data,
      "Inputs to ocv::detect_motion_3frame_differencing are null" );
  }

  if( reset_model )
  {
    d_->reset();
  }

  kv::image_of< uint8_t > const source( image->get_image() );

  return std::make_shared< vital::simple_image_container >(
    vital::image( d_->process_image( source ) ) );
}

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver
