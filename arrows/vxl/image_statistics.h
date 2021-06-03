#ifndef IMAGE_STATISTICS_H_
#define IMAGE_STATISTICS_H_

#include <vil/vil_image_view.h>

#include <vital/range/iota.h>

#include <vil/algo/vil_threshold.h>
#include <vil/vil_image_view.h>
#include <vil/vil_plane.h>

#include <algorithm>
#include <limits>
#include <vector>

// ----------------------------------------------------------------------------
// Calculate the values of our image percentiles from x sampling points
template < typename PixType >
std::vector< PixType >
sample_and_sort_image( vil_image_view< PixType > const& src,
                       unsigned int sampling_points,
                       bool remove_extremes )
{
  if( src.ni() * src.nj() < sampling_points )
  {
    sampling_points = src.ni() * src.nj();
  }

  std::vector< PixType > dst;

  if( sampling_points == 0 )
  {
    return dst;
  }

  auto const scanning_area = src.size();
  auto const ni = src.ni();
  auto const nj = src.nj();
  auto const np = src.nplanes();
  auto const pixel_step =
    static_cast< unsigned >( scanning_area / sampling_points );

  dst.reserve( sampling_points * np );

  unsigned position = 0;

  for( unsigned p = 0; p < np; ++p )
  {
    for( unsigned s = 0; s < sampling_points; ++s, position += pixel_step )
    {
      unsigned i = position % ni;
      unsigned j = ( position / ni ) % nj;
      dst.push_back( src( i, j, p ) );
    }
  }

  std::sort( dst.begin(), dst.end() );

  if( remove_extremes )
  {
    constexpr auto low = PixType{ 0 };

    while( !dst.empty() && dst.front() == low )
    {
      dst.erase( dst.begin() );
    }

    constexpr auto high = std::numeric_limits< PixType >::max();

    while( !dst.empty() && dst.back() == high )
    {
      dst.pop_back();
    }
  }
  return dst;
}

// ----------------------------------------------------------------------------
// Estimate the pixel values at given percentiles using a subset of points
template < typename PixType >
std::vector< PixType >
get_image_percentiles(
  vil_image_view< PixType > const& src,
  std::vector< double > const& percentiles,
  unsigned sampling_points, bool remove_extremes = false )
{
  std::vector< PixType > sorted_samples =
    sample_and_sort_image( src, sampling_points, remove_extremes );

  std::vector< PixType > dst( percentiles.size() );
  double sampling_points_minus1 =
    static_cast< double >( sorted_samples.size() - 1 );

  for( auto const i : kwiver::vital::range::iota( percentiles.size() ) )
  {
    // Find the index by multiplying the number of points by the percentile
    // The number is adjusted by -1 to account for the fact that percentiles
    // are the number which fall below a value. The +0.5 is to account for
    // truncation by static cast.
    auto ind =
      static_cast< size_t >(
        sampling_points_minus1 * percentiles[ i ] + 0.5 );
    dst[ i ] = sorted_samples[ ind ];
  }
  return dst;
}

// ----------------------------------------------------------------------------
// Resultant image is true where pixel is greater than a percentile threshold
template < typename PixType >
void
percentile_threshold_above(
  vil_image_view< PixType > const& src,
  std::vector< double > const& percentiles,
  vil_image_view< bool >& dst, unsigned sampling_points = 1000 )
{
  // Calculate thresholds
  auto thresholds = get_image_percentiles( src, percentiles, sampling_points );
  dst.set_size( src.ni(), src.nj(),
                static_cast< unsigned >( percentiles.size() ) );

  // Perform thresholding
  for( unsigned i = 0; i < thresholds.size(); i++ )
  {
    vil_image_view< bool > output_plane = vil_plane( dst, i );
    vil_threshold_above( src, output_plane, thresholds[ i ] );
  }
}

// ----------------------------------------------------------------------------
// Resultant image is true where pixel is greater than a percentile threshold
template < typename PixType >
void
percentile_threshold_above(
  const vil_image_view< PixType >& src, const double percentile,
  vil_image_view< bool >& dst, unsigned sampling_points = 1000 )
{
  percentile_threshold_above(
    src, std::vector< double >( 1, percentile ), dst, sampling_points );
}

#endif
