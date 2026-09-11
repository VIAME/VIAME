/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief What the viame CSV writer makes of a detection's mask
///
/// `write_detected_object_set_viame_csv` turns a mask into `(poly)` and
/// `(hole)` cells with `cv::findContours` and either `cv::approxPolyDP` or
/// `simplify_polygon`. It is the last OpenCV in `plugins/core` along with
/// the track writer that does the same thing, and it had no test.
///
/// This is the recording that has to exist before it can be ported. The
/// expected output is committed beside this file; regenerate it with
///
///     VIAME_RECORD_MASK_POLYGON_CSV=1 ./tests/bin/test-viame_core-mask_polygon_csv
///
/// which writes the file and fails, so a regeneration is never accidental.
/// Six masks between them reach every branch: a filled disc (one contour), a
/// ring (a contour and a hole), two blobs in one mask (two contours), an L
/// (a contour whose corners the simplifier has to keep), a single pixel (the
/// degenerate case) and an empty mask (no contour at all).

#include <gtest/gtest.h>

#include "write_detected_object_set_viame_csv.h"

#include <viame/core_types/detected_object.h>
#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/image_container.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace kv = kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
std::string
expected_path()
{
#ifdef VIAME_MASK_POLYGON_CSV_DIR
  return std::string( VIAME_MASK_POLYGON_CSV_DIR ) + "/mask_polygon_csv.txt";
#else
  return "mask_polygon_csv.txt";
#endif
}

// ----------------------------------------------------------------------------
/// A mask of \p width by \p height where \p inside decides each pixel.
template < typename Predicate >
kv::image_container_sptr
make_mask( size_t width, size_t height, Predicate inside )
{
  // Eight bit, which is what `convert_polygons_to_mask` and the two python
  // segmenters produce. The bridge refuses a `bool` image outright, so a
  // mask of that type would throw rather than be written.
  kv::image_of< uint8_t > mask( width, height, 1 );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      mask( i, j, 0 ) = inside( i, j ) ? 255 : 0;
    }
  }

  return std::make_shared< kv::simple_image_container >( kv::image( mask ) );
}

// ----------------------------------------------------------------------------
struct named_mask
{
  std::string name;
  kv::bounding_box_d box;
  kv::image_container_sptr mask;
};

// ----------------------------------------------------------------------------
std::vector< named_mask >
masks()
{
  std::vector< named_mask > out;

  auto const disc = []( size_t i, size_t j )
  {
    double const di = static_cast< double >( i ) - 12.0;
    double const dj = static_cast< double >( j ) - 12.0;
    return di * di + dj * dj <= 100.0;
  };

  auto const ring = []( size_t i, size_t j )
  {
    double const di = static_cast< double >( i ) - 15.0;
    double const dj = static_cast< double >( j ) - 15.0;
    double const r2 = di * di + dj * dj;
    return r2 <= 169.0 && r2 >= 36.0;
  };

  auto const blobs = []( size_t i, size_t j )
  {
    bool const first = i >= 2 && i <= 9 && j >= 2 && j <= 9;
    bool const second = i >= 16 && i <= 25 && j >= 12 && j <= 19;
    return first || second;
  };

  auto const ell = []( size_t i, size_t j )
  {
    return ( i < 6 && j < 20 ) || ( j >= 14 && i < 18 );
  };

  auto const dot = []( size_t i, size_t j )
  {
    return i == 4 && j == 5;
  };

  auto const nothing = []( size_t, size_t ) { return false; };

  out.push_back( { "disc", kv::bounding_box_d( 100, 50, 125, 75 ),
                   make_mask( 25, 25, disc ) } );
  out.push_back( { "ring", kv::bounding_box_d( 10, 10, 41, 41 ),
                   make_mask( 31, 31, ring ) } );
  out.push_back( { "blobs", kv::bounding_box_d( 0, 0, 28, 22 ),
                   make_mask( 28, 22, blobs ) } );
  out.push_back( { "ell", kv::bounding_box_d( 7, 3, 27, 23 ),
                   make_mask( 20, 20, ell ) } );
  out.push_back( { "dot", kv::bounding_box_d( 60, 60, 70, 70 ),
                   make_mask( 10, 10, dot ) } );
  out.push_back( { "empty", kv::bounding_box_d( 0, 0, 8, 8 ),
                   make_mask( 8, 8, nothing ) } );

  return out;
}

// ----------------------------------------------------------------------------
/// The writer's output for one mask under one configuration, as the cells
/// that describe the polygon.
std::string
polygon_cells( named_mask const& subject, double tolerance, int points )
{
  auto const path = fs::temp_directory_path() /
    ( "viame_mask_polygon_" + subject.name + ".csv" );

  {
    viame::write_detected_object_set_viame_csv writer;

    auto config = writer.get_configuration();
    config->set_value( "mask_to_poly_tol", tolerance );
    config->set_value( "mask_to_poly_points", points );
    writer.set_configuration( config );

    writer.open( path.string() );

    auto detection = std::make_shared< kv::detected_object >(
      subject.box, 1.0 );
    detection->set_mask( subject.mask );

    auto set = std::make_shared< kv::detected_object_set >();
    set->add( detection );

    writer.write_set( set, "frame.png" );
    writer.close();
  }

  std::ifstream in( path );
  std::string line;
  std::string found;

  while( std::getline( in, line ) )
  {
    if( line.empty() || line[ 0 ] == '#' )
    {
      continue;
    }

    // Everything from the first `(poly)` or `(hole)` cell onwards
    auto const at = line.find( ",(poly)" );
    auto const hole = line.find( ",(hole)" );
    auto const start = std::min( at, hole );

    found = ( start == std::string::npos ) ? "<none>" : line.substr( start + 1 );
    break;
  }

  fs::remove( path );

  return found;
}

// ----------------------------------------------------------------------------
/// Every mask under every configuration, as one text block.
std::string
record()
{
  // The two ways of simplifying, and the disabled case. A negative
  // `mask_to_poly_points` with a non-negative tolerance selects
  // `approxPolyDP`; the other way round selects `simplify_polygon`.
  std::vector< std::pair< double, int > > const settings = {
    { -1.0, 20 },   // the shipped default: at most twenty points
    { -1.0, 6 },    // few enough that the simplifier has to choose
    { -1.0, 1000 }, // more than any contour here has, so nothing is dropped
    { 0.02, -1 },   // a tolerance rather than a count
    { 0.10, -1 },   // a coarse tolerance
  };

  std::ostringstream out;

  for( auto const& subject : masks() )
  {
    for( auto const& setting : settings )
    {
      out << subject.name << " tol=" << setting.first
          << " points=" << setting.second << "\n  "
          << polygon_cells( subject, setting.first, setting.second ) << "\n";
    }
  }

  return out.str();
}

} // namespace

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST ( mask_polygon_csv, matches_the_recording )
{
  auto const found = record();

  if( std::getenv( "VIAME_RECORD_MASK_POLYGON_CSV" ) )
  {
    std::ofstream out( expected_path() );
    out << found;
    FAIL() << "re-recorded " << expected_path() << "; run again without "
              "VIAME_RECORD_MASK_POLYGON_CSV";
  }

  std::ifstream in( expected_path() );
  ASSERT_TRUE( in.good() ) << "could not read " << expected_path();

  std::stringstream buffer;
  buffer << in.rdbuf();

  EXPECT_EQ( found, buffer.str() );
}
