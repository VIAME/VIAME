/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The stereo rig JSON reader and writer, recorded before P8-T06
/// replaced the library underneath them.
///
/// `read_stereo_rig_json` and `write_stereo_rig_json` were cereal's JSON
/// archives, and cereal is what P8-T06 deletes. The reader has a golden case
/// -- `stereo_fish_json` in `tests/golden/calib` -- which says what a real
/// calibration file means. This says the things a golden through
/// `load_stereo_calibration` cannot see:
///
/// * that the reader finds its fields **by name**, skipping the ones it does
///   not want. Every calibration file VIAME writes starts with image size,
///   grid size and residuals, and the reader wants none of them;
/// * that `k3` is optional and a missing one is zero rather than an error,
///   which is the only field with that property;
/// * what the writer's output actually looks like, to the byte. Nothing in
///   this tree calls the writer, so without this there is nothing at all
///   holding its format.

#include <viame/file_io/camera_rig_io.h>

#include <viame/core_types/camera_intrinsics.h>
#include <viame/core_types/camera_perspective.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace kv = kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
class scratch_json
{
public:
  explicit scratch_json( std::string const& name, std::string const& text )
    : m_path( name )
  {
    std::ofstream out( m_path );
    out << text;
  }

  ~scratch_json() { std::remove( m_path.c_str() ); }

  std::string const& path() const { return m_path; }

  std::string contents() const
  {
    std::ifstream in( m_path, std::ios::binary );
    if( !in.is_open() ) { return {}; }
    return std::string( std::istreambuf_iterator< char >( in ),
                        std::istreambuf_iterator< char >() );
  }

private:
  std::string m_path;
};

// ----------------------------------------------------------------------------
// The shape every calibration file VIAME writes has: nine numbers per camera
// that the reader wants, a translation and a rotation, and -- first -- six
// fields it does not.
constexpr char const* const a_calibration = R"({
  "image_width": 1280,
  "image_height": 720,
  "grid_width": 9,
  "grid_height": 6,
  "square_size_mm": 80,
  "rms_error_stereo": 0.5,
  "fx_left": 800.0,
  "fy_left": 801.0,
  "cx_left": 640.0,
  "cy_left": 360.0,
  "k1_left": 0.1,
  "k2_left": 0.2,
  "p1_left": 0.3,
  "p2_left": 0.4,
  "k3_left": 0.5,
  "fx_right": 810.0,
  "fy_right": 811.0,
  "cx_right": 641.0,
  "cy_right": 361.0,
  "k1_right": 0.6,
  "k2_right": 0.7,
  "p1_right": 0.8,
  "p2_right": 0.9,
  "k3_right": 1.0,
  "T": [-200.0, 0.0, 0.0],
  "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
})";

} // namespace

// ----------------------------------------------------------------------------
TEST ( camera_rig_json, the_fields_it_does_not_want_are_skipped )
{
  scratch_json file( "rig_named_fields.json", a_calibration );

  auto const rig = viame::read_stereo_rig_json( file.path() );
  ASSERT_TRUE( rig != nullptr );

  auto const& left = dynamic_cast< kv::camera_perspective const& >(
    *rig->camera( "left" ) );
  auto const& intrinsics = *left.intrinsics();

  // 800, not 1280: the first number in the file is the image width, and a
  // reader that took fields in order rather than by name would have it.
  EXPECT_DOUBLE_EQ( 800.0, intrinsics.focal_length() );
  EXPECT_DOUBLE_EQ( 800.0 / 801.0, intrinsics.aspect_ratio() );
  EXPECT_DOUBLE_EQ( 640.0, intrinsics.principal_point()[ 0 ] );
  EXPECT_DOUBLE_EQ( 360.0, intrinsics.principal_point()[ 1 ] );

  auto const& d = intrinsics.dist_coeffs();
  ASSERT_EQ( 5u, d.size() );
  EXPECT_DOUBLE_EQ( 0.1, d[ 0 ] );
  EXPECT_DOUBLE_EQ( 0.2, d[ 1 ] );
  EXPECT_DOUBLE_EQ( 0.3, d[ 2 ] );
  EXPECT_DOUBLE_EQ( 0.4, d[ 3 ] );
  EXPECT_DOUBLE_EQ( 0.5, d[ 4 ] );
}

// ----------------------------------------------------------------------------
TEST ( camera_rig_json, the_right_camera_is_placed_by_T_and_R )
{
  scratch_json file( "rig_extrinsics.json", a_calibration );

  auto const rig = viame::read_stereo_rig_json( file.path() );
  ASSERT_TRUE( rig != nullptr );

  auto const& left = dynamic_cast< kv::camera_perspective const& >(
    *rig->camera( "left" ) );
  auto const& right = dynamic_cast< kv::camera_perspective const& >(
    *rig->camera( "right" ) );

  // The left camera is the origin, always: the file says nothing about where
  // it is and the reader does not ask.
  EXPECT_DOUBLE_EQ( 0.0, left.center()[ 0 ] );
  EXPECT_DOUBLE_EQ( 0.0, left.center()[ 1 ] );
  EXPECT_DOUBLE_EQ( 0.0, left.center()[ 2 ] );

  EXPECT_DOUBLE_EQ( -200.0, right.translation()[ 0 ] );
  EXPECT_DOUBLE_EQ( 0.0, right.translation()[ 1 ] );
  EXPECT_DOUBLE_EQ( 0.0, right.translation()[ 2 ] );

  auto const rotation = right.rotation().matrix();
  for( unsigned i = 0; i < 3; ++i )
  {
    for( unsigned j = 0; j < 3; ++j )
    {
      EXPECT_DOUBLE_EQ( i == j ? 1.0 : 0.0, rotation( i, j ) )
        << "at " << i << ", " << j;
    }
  }
}

// ----------------------------------------------------------------------------
// k3 alone is read inside a try, so a file written by something that only
// models four coefficients still loads. Every other field is required, and
// the difference is not written down anywhere but here.
TEST ( camera_rig_json, a_missing_k3_is_zero_rather_than_an_error )
{
  std::string text( a_calibration );
  auto const cut = text.find( "  \"k3_right\": 1.0,\n" );
  ASSERT_NE( std::string::npos, cut );
  text.erase( cut, std::string( "  \"k3_right\": 1.0,\n" ).size() );

  scratch_json file( "rig_no_k3.json", text );

  auto const rig = viame::read_stereo_rig_json( file.path() );
  ASSERT_TRUE( rig != nullptr );

  auto const& right = dynamic_cast< kv::camera_perspective const& >(
    *rig->camera( "right" ) );
  auto const& d = right.intrinsics()->dist_coeffs();
  ASSERT_EQ( 5u, d.size() );
  EXPECT_DOUBLE_EQ( 0.9, d[ 3 ] );
  EXPECT_DOUBLE_EQ( 0.0, d[ 4 ] );
}

// ----------------------------------------------------------------------------
// Every field but k3 is required, and a missing one is a throw rather than a
// default. Recorded for the exception's *type* as much as for the fact of
// it: a caller that catches the wrong base catches nothing.
TEST ( camera_rig_json, a_missing_required_field_throws )
{
  std::string text( a_calibration );
  auto const cut = text.find( "  \"fx_left\": 800.0,\n" );
  ASSERT_NE( std::string::npos, cut );
  text.erase( cut, std::string( "  \"fx_left\": 800.0,\n" ).size() );

  scratch_json file( "rig_no_fx.json", text );

  EXPECT_THROW( viame::read_stereo_rig_json( file.path() ),
                std::runtime_error );
}

// ----------------------------------------------------------------------------
// A file that is not JSON at all fails the same way, rather than returning a
// rig of zeroes.
TEST ( camera_rig_json, a_file_that_is_not_json_throws )
{
  scratch_json file( "rig_not_json.json", "this is not json" );

  EXPECT_THROW( viame::read_stereo_rig_json( file.path() ),
                std::runtime_error );
}

// ----------------------------------------------------------------------------
// The writer's whole output, because nothing else holds it: no applet, no
// pipeline and no test calls `write_stereo_rig`, so its format is whatever
// this says it is.
//
// Two-space indent, one array element per line, and no trailing newline.
// Numbers are the shortest text that reads back as the same double, which is
// why a value that went in as `0.1` comes out as `0.1` and the rotation
// comes out at seventeen digits.
TEST ( camera_rig_json, the_writer_writes_this )
{
  scratch_json source( "rig_writer_source.json", a_calibration );
  auto const rig = viame::read_stereo_rig_json( source.path() );

  scratch_json written( "rig_writer_output.json", "" );
  viame::write_stereo_rig( rig, written.path() );

  EXPECT_EQ(
    "{\n"
    "  \"fx_left\": 800.0,\n"
    "  \"fy_left\": 801.0,\n"
    "  \"cx_left\": 640.0,\n"
    "  \"cy_left\": 360.0,\n"
    "  \"k1_left\": 0.1,\n"
    "  \"k2_left\": 0.2,\n"
    "  \"p1_left\": 0.3,\n"
    "  \"p2_left\": 0.4,\n"
    "  \"k3_left\": 0.5,\n"
    "  \"fx_right\": 810.0,\n"
    "  \"fy_right\": 811.0,\n"
    "  \"cx_right\": 641.0,\n"
    "  \"cy_right\": 361.0,\n"
    "  \"k1_right\": 0.6,\n"
    "  \"k2_right\": 0.7,\n"
    "  \"p1_right\": 0.8,\n"
    "  \"p2_right\": 0.9,\n"
    "  \"k3_right\": 1.0,\n"
    // Negative zero, and it is not a typo. The right camera's translation
    // is computed as `t - R c` with `c` the left camera's centre at the
    // origin, and the subtraction gives -0.0 where the input said 0.0. It
    // reads back as zero and compares equal to zero, so nothing downstream
    // notices -- but it is in the file, and a writer that normalised it away
    // would be writing different bytes.
    "  \"T\": [\n"
    "    -200.0,\n"
    "    -0.0,\n"
    "    -0.0\n"
    "  ],\n"
    "  \"R\": [\n"
    "    1.0,\n"
    "    0.0,\n"
    "    0.0,\n"
    "    0.0,\n"
    "    1.0,\n"
    "    0.0,\n"
    "    0.0,\n"
    "    0.0,\n"
    "    1.0\n"
    "  ]\n"
    "}",
    written.contents() );
}

// ----------------------------------------------------------------------------
// The extra fields do not survive: what the writer emits is what the reader
// asked for and nothing else. A file that goes through both loses its image
// size, its grid and its residuals -- which is a thing to know before
// pointing the writer at a file someone else will read.
TEST ( camera_rig_json, a_round_trip_keeps_the_numbers_and_drops_the_rest )
{
  scratch_json source( "rig_round_trip_source.json", a_calibration );
  auto const first = viame::read_stereo_rig_json( source.path() );

  scratch_json written( "rig_round_trip.json", "" );
  viame::write_stereo_rig( first, written.path() );

  auto const second = viame::read_stereo_rig_json( written.path() );
  ASSERT_TRUE( second != nullptr );

  EXPECT_EQ( std::string::npos, written.contents().find( "image_width" ) );

  for( auto const& name : { std::string( "left" ), std::string( "right" ) } )
  {
    auto const& before = dynamic_cast< kv::camera_perspective const& >(
      *first->camera( name ) );
    auto const& after = dynamic_cast< kv::camera_perspective const& >(
      *second->camera( name ) );

    EXPECT_DOUBLE_EQ( before.intrinsics()->focal_length(),
                      after.intrinsics()->focal_length() );
    EXPECT_DOUBLE_EQ( before.intrinsics()->aspect_ratio(),
                      after.intrinsics()->aspect_ratio() );

    auto const& d1 = before.intrinsics()->dist_coeffs();
    auto const& d2 = after.intrinsics()->dist_coeffs();
    ASSERT_EQ( d1.size(), d2.size() );
    for( unsigned i = 0; i < d1.size(); ++i )
    {
      EXPECT_DOUBLE_EQ( d1[ i ], d2[ i ] ) << "coefficient " << i;
    }
  }
}

// ----------------------------------------------------------------------------
// `write_stereo_rig` dispatches on the extension and silently does nothing
// for one it does not handle. `.yml` is a stub with a TODO in it; anything
// else is not dispatched at all.
TEST ( camera_rig_json, an_extension_it_does_not_write_writes_nothing )
{
  scratch_json source( "rig_extension_source.json", a_calibration );
  auto const rig = viame::read_stereo_rig_json( source.path() );

  scratch_json yaml( "rig_extension.yml", "" );
  scratch_json other( "rig_extension.txt", "" );

  viame::write_stereo_rig( rig, yaml.path() );
  viame::write_stereo_rig( rig, other.path() );

  EXPECT_TRUE( yaml.contents().empty() );
  EXPECT_TRUE( other.contents().empty() );
}
