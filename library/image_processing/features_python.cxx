/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Python bindings for the ported feature detectors
///
/// SIFT and SURF are C++ on this branch -- `sift.h` and `surf.h` -- and were
/// reachable only through the algorithm framework, as `detect_features:ocv_SIFT`
/// and `ocv_SURF`. That is the right interface for a pipeline and the wrong one
/// for a script: the registration utilities, the multimodal registration, the
/// homography IOU tracker and colmap's reconstruction all want a detector as a
/// function of an array, which is what `cv2.SIFT_create().detectAndCompute` gave
/// them and what kept them importing cv2.
///
/// Keypoints come back as one **(n, 6) float32 array** -- x, y, size, angle,
/// response, and the packed octave -- rather than as a list of objects, because
/// every caller either reads a column of them or passes the lot to a matcher.
/// The packed octave is carried through so that describing a detected keypoint
/// samples the level of the pyramid it was found at; a caller inventing
/// keypoints of its own leaves it at zero, which describes them at the base.

#include <viame/image_processing/sift.h>
#include <viame/image_processing/surf.h>

#include <viame/core_types/image.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using array_u8 = py::array_t< uint8_t, py::array::c_style >;
using array_f = py::array_t< float, py::array::c_style | py::array::forcecast >;

/// One plane or three, as `image_of< uint8_t >`.
viame::image_of< uint8_t >
as_image( array_u8 const& array, char const* who )
{
  auto const buffer = array.request();

  if( buffer.ndim != 2 && buffer.ndim != 3 )
  {
    throw std::invalid_argument(
      std::string( who ) + " wants a two or three dimensional array" );
  }

  auto const height = static_cast< size_t >( buffer.shape[ 0 ] );
  auto const width = static_cast< size_t >( buffer.shape[ 1 ] );
  auto const planes = buffer.ndim == 3
                      ? static_cast< size_t >( buffer.shape[ 2 ] ) : size_t{ 1 };

  if( planes != 1 && planes != 3 )
  {
    throw std::invalid_argument(
      std::string( who ) + " wants one or three planes" );
  }

  auto const* data = static_cast< uint8_t const* >( buffer.ptr );
  viame::image_of< uint8_t > out( width, height, planes );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      for( size_t p = 0; p < planes; ++p )
      {
        out( i, j, p ) = data[ ( j * width + i ) * planes + p ];
      }
    }
  }

  return out;
}

/// The six columns, in the order `features.py` documents.
template < typename Keypoint >
py::array_t< float >
as_keypoint_array( std::vector< Keypoint > const& keypoints, bool has_octave )
{
  auto const count = keypoints.size();
  py::array_t< float > out(
    { static_cast< py::ssize_t >( count ), py::ssize_t{ 6 } } );
  auto view = out.mutable_unchecked< 2 >();

  for( size_t k = 0; k < count; ++k )
  {
    auto const row = static_cast< py::ssize_t >( k );
    view( row, 0 ) = keypoints[ k ].x;
    view( row, 1 ) = keypoints[ k ].y;
    view( row, 2 ) = keypoints[ k ].size;
    view( row, 3 ) = keypoints[ k ].angle;
    view( row, 4 ) = keypoints[ k ].response;
    view( row, 5 ) = has_octave
                     ? static_cast< float >( keypoints[ k ].octave ) : 0.0f;
  }

  return out;
}

py::array_t< float >
as_descriptor_array( std::vector< float > const& raw, int width )
{
  auto const count = width > 0
                     ? raw.size() / static_cast< size_t >( width ) : size_t{ 0 };
  py::array_t< float > out(
    { static_cast< py::ssize_t >( count ),
      static_cast< py::ssize_t >( width ) } );

  if( count )
  {
    std::copy( raw.begin(), raw.end(), out.mutable_data() );
  }

  return out;
}

std::vector< viame::sift::keypoint >
as_sift_keypoints( array_f const& array )
{
  auto const buffer = array.request();

  if( buffer.ndim != 2 || buffer.shape[ 1 ] < 5 )
  {
    throw std::invalid_argument(
      "sift_describe wants an (n, 5) or (n, 6) keypoint array" );
  }

  auto const count = static_cast< size_t >( buffer.shape[ 0 ] );
  auto const stride = static_cast< size_t >( buffer.shape[ 1 ] );
  auto const* data = static_cast< float const* >( buffer.ptr );

  std::vector< viame::sift::keypoint > out( count );

  for( size_t k = 0; k < count; ++k )
  {
    out[ k ].x = data[ k * stride + 0 ];
    out[ k ].y = data[ k * stride + 1 ];
    out[ k ].size = data[ k * stride + 2 ];
    out[ k ].angle = data[ k * stride + 3 ];
    out[ k ].response = data[ k * stride + 4 ];
    out[ k ].octave = stride > 5
                      ? static_cast< int >( data[ k * stride + 5 ] ) : 0;
  }

  return out;
}

viame::sift::settings
sift_settings( int n_features, int n_octave_layers, double contrast_threshold,
               double edge_threshold, double sigma )
{
  viame::sift::settings settings;
  settings.n_features = n_features;
  settings.n_octave_layers = n_octave_layers;
  settings.contrast_threshold = contrast_threshold;
  settings.edge_threshold = edge_threshold;
  settings.sigma = sigma;

  return settings;
}

py::tuple
sift_detect_and_compute( array_u8 const& array, int n_features,
                         int n_octave_layers, double contrast_threshold,
                         double edge_threshold, double sigma,
                         bool describe )
{
  auto const image = as_image( array, "sift" );
  auto const settings = sift_settings( n_features, n_octave_layers,
                                       contrast_threshold, edge_threshold,
                                       sigma );

  std::vector< viame::sift::keypoint > keypoints;
  std::vector< float > descriptors;

  {
    py::gil_scoped_release release;
    viame::sift::detect_and_compute( image, settings, keypoints,
                                   describe ? &descriptors : nullptr );
  }

  return py::make_tuple(
    as_keypoint_array( keypoints, true ),
    describe ? as_descriptor_array( descriptors,
                                    viame::sift::descriptor_size() )
             : as_descriptor_array( {}, viame::sift::descriptor_size() ) );
}

py::tuple
sift_describe( array_u8 const& array, array_f const& keypoint_array,
               int n_features, int n_octave_layers, double contrast_threshold,
               double edge_threshold, double sigma )
{
  auto const image = as_image( array, "sift_describe" );
  auto const settings = sift_settings( n_features, n_octave_layers,
                                       contrast_threshold, edge_threshold,
                                       sigma );

  auto keypoints = as_sift_keypoints( keypoint_array );
  std::vector< float > descriptors;

  {
    py::gil_scoped_release release;
    viame::sift::detect_and_compute( image, settings, keypoints, &descriptors,
                                   true );
  }

  return py::make_tuple(
    as_keypoint_array( keypoints, true ),
    as_descriptor_array( descriptors, viame::sift::descriptor_size() ) );
}

py::tuple
surf_detect_and_compute( array_u8 const& array, double hessian_threshold,
                         int n_octaves, int n_octaves_layers, bool extended,
                         bool upright, bool describe )
{
  auto const image = as_image( array, "surf" );

  viame::surf::settings settings;
  settings.hessian_threshold = hessian_threshold;
  settings.n_octaves = n_octaves;
  settings.n_octaves_layers = n_octaves_layers;
  settings.extended = extended;
  settings.upright = upright;

  std::vector< viame::surf::keypoint > keypoints;
  std::vector< float > descriptors;

  {
    py::gil_scoped_release release;
    viame::surf::detect_and_compute( image, settings, keypoints,
                                   describe ? &descriptors : nullptr );
  }

  auto const width = viame::surf::descriptor_size( settings );

  return py::make_tuple(
    as_keypoint_array( keypoints, false ),
    describe ? as_descriptor_array( descriptors, width )
             : as_descriptor_array( {}, width ) );
}

// Exact Hamming search with O(query_count * k) output storage. Descriptors
// remain packed; no query-by-train-by-bit comparison array is constructed.
unsigned bit_count( uint64_t value )
{
#if defined(__GNUC__) || defined(__clang__)
  return static_cast< unsigned >( __builtin_popcountll( value ) );
#else
  value -= ( value >> 1 ) & UINT64_C(0x5555555555555555);
  value = ( value & UINT64_C(0x3333333333333333) ) +
          ( ( value >> 2 ) & UINT64_C(0x3333333333333333) );
  value = ( value + ( value >> 4 ) ) & UINT64_C(0x0f0f0f0f0f0f0f0f);
  return static_cast< unsigned >(
    ( value * UINT64_C(0x0101010101010101) ) >> 56 );
#endif
}

py::tuple nearest_binary( array_u8 const& query, array_u8 const& train, int k )
{
  if( query.ndim() != 2 || train.ndim() != 2 ||
      query.shape( 1 ) != train.shape( 1 ) || k < 1 || k > train.shape( 0 ) )
  {
    throw std::invalid_argument( "nearest_binary: invalid descriptor shapes or k" );
  }
  auto const nq = query.shape( 0 ), nt = train.shape( 0 );
  auto const width = static_cast< size_t >( query.shape( 1 ) );
  py::array_t< int64_t > indices( { nq, py::ssize_t{ k } } );
  py::array_t< int > distances( { nq, py::ssize_t{ k } } );
  auto* out_index = indices.mutable_data();
  auto* out_distance = distances.mutable_data();
  auto const* q = query.data();
  auto const* t = train.data();
  {
    py::gil_scoped_release release;
    for( py::ssize_t i = 0; i < nq; ++i )
    {
      auto* best = out_distance + i * k;
      auto* found = out_index + i * k;
      std::fill( best, best + k, std::numeric_limits< int >::max() );
      std::fill( found, found + k, int64_t{ -1 } );
      for( py::ssize_t j = 0; j < nt; ++j )
      {
        int distance = 0;
        size_t b = 0;
        for( ; b + sizeof( uint64_t ) <= width; b += sizeof( uint64_t ) )
        {
          uint64_t a, c;
          std::memcpy( &a, q + i * width + b, sizeof( a ) );
          std::memcpy( &c, t + j * width + b, sizeof( c ) );
          distance += bit_count( a ^ c );
        }
        for( ; b < width; ++b )
        {
          distance += bit_count( q[ i * width + b ] ^ t[ j * width + b ] );
        }
        // Train rows are visited in ascending order; strict improvement
        // preserves the lower train index on ties, including the kth tie.
        if( distance >= best[ k - 1 ] ) { continue; }
        int at = k - 1;
        while( at > 0 && distance < best[ at - 1 ] )
        {
          best[ at ] = best[ at - 1 ];
          found[ at ] = found[ at - 1 ];
          --at;
        }
        best[ at ] = distance;
        found[ at ] = j;
      }
    }
  }
  return py::make_tuple( indices, distances );
}

} // namespace

PYBIND11_MODULE( _features, m )
{
  m.def( "nearest_binary", &nearest_binary );

  m.doc() = "VIAME's own SIFT and SURF, as arrays rather than as algorithms";

  m.def( "sift", &sift_detect_and_compute, py::arg( "image" ),
         py::arg( "n_features" ) = 0, py::arg( "n_octave_layers" ) = 3,
         py::arg( "contrast_threshold" ) = 0.04,
         py::arg( "edge_threshold" ) = 10.0, py::arg( "sigma" ) = 1.6,
         py::arg( "describe" ) = true,
         "Detect SIFT keypoints and describe them. Returns "
         "(keypoints, descriptors): an (n, 6) array of x, y, size, angle, "
         "response and packed octave, and an (n, 128) array of descriptors on "
         "a 0..255 scale. `cv2.SIFT_create( ... ).detectAndCompute`, with the "
         "arguments in the order cv2 takes them." );

  m.def( "sift_describe", &sift_describe, py::arg( "image" ),
         py::arg( "keypoints" ), py::arg( "n_features" ) = 0,
         py::arg( "n_octave_layers" ) = 3,
         py::arg( "contrast_threshold" ) = 0.04,
         py::arg( "edge_threshold" ) = 10.0, py::arg( "sigma" ) = 1.6,
         "Describe the given keypoints, which is `SIFT.compute`. Returns the "
         "keypoints as well, because a keypoint too near a border to describe "
         "is dropped from both together and a caller holding the old array "
         "would pair every descriptor after the first drop with the wrong "
         "keypoint." );

  m.def( "surf", &surf_detect_and_compute, py::arg( "image" ),
         py::arg( "hessian_threshold" ) = 100.0, py::arg( "n_octaves" ) = 4,
         py::arg( "n_octaves_layers" ) = 3, py::arg( "extended" ) = false,
         py::arg( "upright" ) = false, py::arg( "describe" ) = true,
         "Detect SURF keypoints and describe them, as "
         "`cv2.xfeatures2d.SURF_create( ... ).detectAndCompute` would if any "
         "wheel carried it. The keypoint array's sixth column is zero: SURF "
         "has no packed octave." );
}
