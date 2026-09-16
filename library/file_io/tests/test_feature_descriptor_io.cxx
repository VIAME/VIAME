/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The KWFD file format, recorded before P8-T06 replaced the library
/// that wrote it.
///
/// `feature_descriptor_io` wrote its files with cereal's
/// `PortableBinaryArchive`, and cereal is what P8-T06 deletes. Nothing in
/// this tree reads a KWFD file that this tree did not write, so a rewrite
/// that changed the layout would have passed every test there was and made
/// every file anyone already had unreadable.
///
/// So the layout is written down here as bytes, taken by running the cereal
/// implementation, before anything replaced it. The expectations are a
/// recording and not an opinion: the endianness flag in the middle of the
/// header, the 64-bit counts, and the fact that a 2x2 covariance is three
/// numbers rather than four are all cereal's decisions, and they are the
/// format now.
///
/// The format, in full:
///
///     "KWFD"            4 bytes, the magic
///     endian flag       1 byte, 1 when the numbers that follow are
///                       little-endian
///     version           uint16, must be 1
///     feature count     uint64
///     feature type      uint8, present only when the count is non-zero
///     features          count x ( 8 numbers then 3 bytes of colour )
///     descriptor count  uint64
///     descriptor type   uint8, present only when the count is non-zero
///     dimension         uint64, present only when the count is non-zero
///     descriptors       count x dimension numbers
///
/// The eight numbers of a feature are x, y, magnitude, scale, angle and the
/// three distinct entries of the covariance; the three bytes are R, G and B.

#include <feature_descriptor_io.h>

#include <viame/core_types/descriptor.h>
#include <viame/core_types/descriptor_set.h>
#include <viame/core_types/feature.h>
#include <viame/core_types/feature_set.h>

#include <viame/algorithm_framework/exceptions.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace kv = viame;
namespace core = viame::core;

namespace {

// ----------------------------------------------------------------------------
// A path in the directory the test was told to work in, removed when the
// fixture goes away so that a failure leaves nothing behind for the next run
// to read by accident.
class scratch_file
{
public:
  explicit scratch_file( std::string const& name )
    : m_path( name )
  {
    std::remove( m_path.c_str() );
  }

  ~scratch_file() { std::remove( m_path.c_str() ); }

  std::string const& path() const { return m_path; }

  // The whole file as bytes. Missing is empty rather than an error, because
  // one of the recordings is that nothing was written.
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
std::string
to_hex( std::string const& bytes )
{
  static char const* const digits = "0123456789abcdef";

  std::string out;
  out.reserve( bytes.size() * 2 );
  for( unsigned char const c : bytes )
  {
    out.push_back( digits[ c >> 4 ] );
    out.push_back( digits[ c & 0xf ] );
  }
  return out;
}

// ----------------------------------------------------------------------------
kv::feature_sptr
a_double_feature()
{
  auto f = std::make_shared< kv::feature_< double > >();
  f->set_loc( kv::vector_2d( 1.0, 2.0 ) );
  f->set_magnitude( 3.0 );
  f->set_scale( 4.0 );
  f->set_angle( 5.0 );

  kv::covariance_2d covar;
  covar( 0, 0 ) = 6.0;
  covar( 0, 1 ) = 7.0;
  covar( 1, 1 ) = 8.0;
  f->set_covar( covar );

  f->set_color( kv::rgb_color( 9, 10, 11 ) );
  return f;
}

// ----------------------------------------------------------------------------
kv::feature_set_sptr
one_double_feature()
{
  return std::make_shared< kv::simple_feature_set >(
    std::vector< kv::feature_sptr >{ a_double_feature() } );
}

// ----------------------------------------------------------------------------
// Four elements rather than 64 or 128, so that the bytes fit on a page. The
// implementation allocates a `descriptor_fixed` for those two sizes and a
// `descriptor_dynamic` for everything else; the file says nothing about
// which, so this is the same format either way -- which is itself one of the
// things recorded below.
kv::descriptor_set_sptr
two_float_descriptors()
{
  std::vector< kv::descriptor_sptr > out;
  for( int d = 0; d < 2; ++d )
  {
    auto desc = std::make_shared< kv::descriptor_dynamic< float > >( 4 );
    float* raw = desc->raw_data();
    for( int i = 0; i < 4; ++i ) { raw[ i ] = static_cast< float >( d * 4 + i ); }
    out.push_back( desc );
  }
  return std::make_shared< kv::simple_descriptor_set >( out );
}

} // namespace

// ----------------------------------------------------------------------------
TEST ( feature_descriptor_io, the_header_is_magic_endianness_and_a_version )
{
  scratch_file file( "fdio_header.kwfd" );

  core::feature_descriptor_io io;
  io.save( file.path(), one_double_feature(), nullptr );

  auto const bytes = file.contents();
  ASSERT_GE( bytes.size(), 7u );

  // "KWFD", then cereal's endianness byte, then the version as a 16-bit
  // number. The endianness byte belongs to cereal's archive rather than to
  // KWIVER's header, which is why it sits between the magic and the version
  // rather than before or after both.
  EXPECT_EQ( "4b574644" "01" "0100", to_hex( bytes.substr( 0, 7 ) ) );
}

// ----------------------------------------------------------------------------
TEST ( feature_descriptor_io, a_double_feature_is_eight_numbers_and_a_colour )
{
  scratch_file file( "fdio_one_feature.kwfd" );

  core::feature_descriptor_io io;
  io.save( file.path(), one_double_feature(), nullptr );

  // 4 magic + 1 endian + 2 version + 8 count + 1 type
  //   + ( 8 x 8 ) + 3 colour + 8 descriptor count
  EXPECT_EQ( 91u, file.contents().size() );

  EXPECT_EQ(
    "4b574644"                              // KWFD
    "01"                                    // little-endian
    "0100"                                  // version 1
    "0100000000000000"                      // one feature
    "13"                                    // double: integer 0, signed 1,
                                            // log2( 8 ) = 3
    "000000000000f03f"                      // x = 1
    "0000000000000040"                      // y = 2
    "0000000000000840"                      // magnitude = 3
    "0000000000001040"                      // scale = 4
    "0000000000001440"                      // angle = 5
    "0000000000001840"                      // covariance ( 0, 0 ) = 6
    "0000000000001c40"                      // covariance ( 0, 1 ) = 7
    "0000000000002040"                      // covariance ( 1, 1 ) = 8
    "090a0b"                                // r, g, b
    "0000000000000000",                     // no descriptors
    to_hex( file.contents() ) );
}

// ----------------------------------------------------------------------------
// The type byte is computed rather than assigned: bit 5 is "is an integer",
// bit 4 is "is signed", and the low bits are log2 of the size. A reader that
// hard-codes the values has the same table; a reader that recomputes them
// gets the same answer.
TEST ( feature_descriptor_io, the_type_byte_encodes_the_type_rather_than_naming_it )
{
  scratch_file doubles( "fdio_type_double.kwfd" );
  scratch_file floats( "fdio_type_float.kwfd" );

  core::feature_descriptor_io as_written;
  as_written.save( doubles.path(), one_double_feature(), nullptr );

  // `write_float_features` narrows on the way out, which is the only reason
  // the type byte is not simply the type of what was passed in.
  core::feature_descriptor_io as_floats( true );
  as_floats.save( floats.path(), one_double_feature(), nullptr );

  EXPECT_EQ( "13", to_hex( doubles.contents().substr( 15, 1 ) ) );
  EXPECT_EQ( "12", to_hex( floats.contents().substr( 15, 1 ) ) );

  // Eight floats instead of eight doubles, and nothing else changes.
  EXPECT_EQ( 91u, doubles.contents().size() );
  EXPECT_EQ( 59u, floats.contents().size() );
}

// ----------------------------------------------------------------------------
TEST ( feature_descriptor_io, descriptors_carry_their_dimension_once )
{
  scratch_file file( "fdio_descriptors.kwfd" );

  core::feature_descriptor_io io;
  io.save( file.path(), nullptr, two_float_descriptors() );

  EXPECT_EQ(
    "4b574644"                              // KWFD
    "01"                                    // little-endian
    "0100"                                  // version 1
    "0000000000000000"                      // no features
    "0200000000000000"                      // two descriptors
    "12"                                    // float
    "0400000000000000"                      // four elements each, written once
    "00000000" "0000803f" "00000040" "00004040"
    "00008040" "0000a040" "0000c040" "0000e040",
    to_hex( file.contents() ) );
}

// ----------------------------------------------------------------------------
// Both counts are always present. A file with neither features nor
// descriptors is never written at all, but a file with one of the two still
// says the other is zero.
TEST ( feature_descriptor_io, an_absent_set_is_a_zero_count_not_an_omission )
{
  scratch_file features_only( "fdio_features_only.kwfd" );
  scratch_file descriptors_only( "fdio_descriptors_only.kwfd" );

  core::feature_descriptor_io io;
  io.save( features_only.path(), one_double_feature(), nullptr );
  io.save( descriptors_only.path(), nullptr, two_float_descriptors() );

  // The last eight bytes of a features-only file are the descriptor count.
  auto const tail = features_only.contents().substr(
    features_only.contents().size() - 8 );
  EXPECT_EQ( "0000000000000000", to_hex( tail ) );

  // And the feature count sits where it always does in a descriptors-only
  // one.
  EXPECT_EQ(
    "0000000000000000",
    to_hex( descriptors_only.contents().substr( 7, 8 ) ) );
}

// ----------------------------------------------------------------------------
TEST ( feature_descriptor_io, nothing_to_write_writes_nothing )
{
  scratch_file file( "fdio_empty.kwfd" );

  core::feature_descriptor_io io;

  // Two null sets is refused by the interface, before the implementation is
  // reached at all.
  EXPECT_THROW( io.save( file.path(), nullptr, nullptr ), kv::invalid_value );
  EXPECT_TRUE( file.contents().empty() );

  // Two *empty* sets is not: the interface is satisfied and the
  // implementation declines to write. Not an empty file -- no file, so a
  // stale file at that path survives. Worth knowing rather than worth
  // changing here.
  auto const empty_features =
    std::make_shared< kv::simple_feature_set >(
      std::vector< kv::feature_sptr >{} );
  auto const empty_descriptors =
    std::make_shared< kv::simple_descriptor_set >(
      std::vector< kv::descriptor_sptr >{} );
  io.save( file.path(), empty_features, empty_descriptors );
  EXPECT_TRUE( file.contents().empty() );
}

// ----------------------------------------------------------------------------
TEST ( feature_descriptor_io, features_survive_the_round_trip )
{
  scratch_file file( "fdio_round_trip_features.kwfd" );

  core::feature_descriptor_io io;
  io.save( file.path(), one_double_feature(), nullptr );

  kv::feature_set_sptr feat;
  kv::descriptor_set_sptr desc;
  io.load( file.path(), feat, desc );

  ASSERT_TRUE( feat != nullptr );
  ASSERT_EQ( 1u, feat->size() );
  EXPECT_EQ( nullptr, desc );

  // By value: `features()` returns the vector, and a reference to an element
  // of it would outlive the temporary.
  auto const f = feat->features()[ 0 ];
  EXPECT_EQ( 1.0, f->loc()[ 0 ] );
  EXPECT_EQ( 2.0, f->loc()[ 1 ] );
  EXPECT_EQ( 3.0, f->magnitude() );
  EXPECT_EQ( 4.0, f->scale() );
  EXPECT_EQ( 5.0, f->angle() );
  EXPECT_EQ( 6.0, f->covar()( 0, 0 ) );
  EXPECT_EQ( 7.0, f->covar()( 0, 1 ) );
  // The lower triangle is the upper one: a 2x2 covariance is three numbers.
  EXPECT_EQ( 7.0, f->covar()( 1, 0 ) );
  EXPECT_EQ( 8.0, f->covar()( 1, 1 ) );
  EXPECT_EQ( 9, f->color().r );
  EXPECT_EQ( 10, f->color().g );
  EXPECT_EQ( 11, f->color().b );

  // Double in, double out.
  EXPECT_EQ( typeid( double ), f->data_type() );
}

// ----------------------------------------------------------------------------
TEST ( feature_descriptor_io, descriptors_survive_the_round_trip )
{
  scratch_file file( "fdio_round_trip_descriptors.kwfd" );

  core::feature_descriptor_io io;
  io.save( file.path(), nullptr, two_float_descriptors() );

  kv::feature_set_sptr feat;
  kv::descriptor_set_sptr desc;
  io.load( file.path(), feat, desc );

  EXPECT_EQ( nullptr, feat );
  ASSERT_TRUE( desc != nullptr );
  ASSERT_EQ( 2u, desc->size() );

  for( size_t d = 0; d < 2; ++d )
  {
    auto const values = desc->at( d )->as_double();
    ASSERT_EQ( 4u, values.size() );
    for( size_t i = 0; i < 4; ++i )
    {
      EXPECT_EQ( static_cast< double >( d * 4 + i ), values[ i ] );
    }
  }

  EXPECT_EQ( typeid( float ), desc->at( 0 )->data_type() );
}

// ----------------------------------------------------------------------------
TEST ( feature_descriptor_io, both_sets_in_one_file )
{
  scratch_file file( "fdio_both.kwfd" );

  core::feature_descriptor_io io;
  io.save( file.path(), one_double_feature(), two_float_descriptors() );

  kv::feature_set_sptr feat;
  kv::descriptor_set_sptr desc;
  io.load( file.path(), feat, desc );

  ASSERT_TRUE( feat != nullptr );
  ASSERT_TRUE( desc != nullptr );
  EXPECT_EQ( 1u, feat->size() );
  EXPECT_EQ( 2u, desc->size() );
}

// ----------------------------------------------------------------------------
// Narrowing is lossy and the file does not say it happened, so a reader gets
// floats back and no warning. Recorded because it is the one case where what
// comes out is not what went in.
TEST ( feature_descriptor_io, write_float_features_narrows_on_the_way_out )
{
  scratch_file file( "fdio_narrowed.kwfd" );

  auto f = std::make_shared< kv::feature_< double > >();
  f->set_loc( kv::vector_2d( 0.1, 0.2 ) );
  auto const features = std::make_shared< kv::simple_feature_set >(
    std::vector< kv::feature_sptr >{ f } );

  core::feature_descriptor_io io( true );
  io.save( file.path(), features, nullptr );

  kv::feature_set_sptr read;
  kv::descriptor_set_sptr desc;
  io.load( file.path(), read, desc );

  ASSERT_TRUE( read != nullptr );
  auto const back = read->features()[ 0 ];
  EXPECT_EQ( typeid( float ), back->data_type() );
  EXPECT_NE( 0.1, back->loc()[ 0 ] );
  EXPECT_FLOAT_EQ( 0.1f, static_cast< float >( back->loc()[ 0 ] ) );
}

// ----------------------------------------------------------------------------
TEST ( feature_descriptor_io, a_file_that_is_not_a_kwfd_file_is_refused )
{
  scratch_file file( "fdio_not_kwfd.kwfd" );

  {
    std::ofstream out( file.path(), std::ios::binary );
    out << "NOPE" << std::string( 32, '\0' );
  }

  core::feature_descriptor_io io;
  kv::feature_set_sptr feat;
  kv::descriptor_set_sptr desc;
  EXPECT_THROW( io.load( file.path(), feat, desc ), kv::invalid_data );
}

// ----------------------------------------------------------------------------
TEST ( feature_descriptor_io, a_version_other_than_one_is_refused )
{
  scratch_file file( "fdio_version_2.kwfd" );

  core::feature_descriptor_io io;
  io.save( file.path(), one_double_feature(), nullptr );

  // Version is the two bytes after the magic and the endianness flag.
  auto bytes = file.contents();
  bytes[ 5 ] = 2;
  {
    std::ofstream out( file.path(), std::ios::binary );
    out.write( bytes.data(), static_cast< std::streamsize >( bytes.size() ) );
  }

  kv::feature_set_sptr feat;
  kv::descriptor_set_sptr desc;
  EXPECT_THROW( io.load( file.path(), feat, desc ), kv::invalid_data );
}

// ----------------------------------------------------------------------------
TEST ( feature_descriptor_io, an_unknown_type_byte_is_refused )
{
  scratch_file file( "fdio_bad_type.kwfd" );

  core::feature_descriptor_io io;
  io.save( file.path(), one_double_feature(), nullptr );

  auto bytes = file.contents();
  bytes[ 15 ] = 0x7f;
  {
    std::ofstream out( file.path(), std::ios::binary );
    out.write( bytes.data(), static_cast< std::streamsize >( bytes.size() ) );
  }

  kv::feature_set_sptr feat;
  kv::descriptor_set_sptr desc;
  EXPECT_THROW( io.load( file.path(), feat, desc ), kv::invalid_data );
}

// ----------------------------------------------------------------------------
TEST ( feature_descriptor_io, a_null_feature_is_refused_rather_than_skipped )
{
  scratch_file file( "fdio_null_feature.kwfd" );

  auto const features = std::make_shared< kv::simple_feature_set >(
    std::vector< kv::feature_sptr >{ a_double_feature(), nullptr } );

  core::feature_descriptor_io io;
  EXPECT_THROW( io.save( file.path(), features, nullptr ), kv::invalid_data );
}

// ----------------------------------------------------------------------------
// The dimension is written once, before the first descriptor, so a set whose
// members disagree cannot be written at all.
TEST ( feature_descriptor_io, descriptors_of_differing_length_are_refused )
{
  scratch_file file( "fdio_ragged.kwfd" );

  std::vector< kv::descriptor_sptr > ragged{
    std::make_shared< kv::descriptor_dynamic< float > >( 4 ),
    std::make_shared< kv::descriptor_dynamic< float > >( 5 ) };

  core::feature_descriptor_io io;
  EXPECT_THROW(
    io.save(
      file.path(), nullptr,
      std::make_shared< kv::simple_descriptor_set >( ragged ) ),
    kv::invalid_data );
}

// ----------------------------------------------------------------------------
// 64 and 128 get a fixed-size allocation on the way back in and everything
// else a dynamic one. The file is identical either way, which is what makes
// that an implementation detail rather than part of the format.
TEST ( feature_descriptor_io, the_common_dimensions_are_not_a_different_format )
{
  scratch_file file( "fdio_dim_128.kwfd" );

  std::vector< kv::descriptor_sptr > descriptors;
  auto d = std::make_shared< kv::descriptor_dynamic< float > >( 128 );
  float* raw = d->raw_data();
  for( int i = 0; i < 128; ++i ) { raw[ i ] = static_cast< float >( i ); }
  descriptors.push_back( d );

  core::feature_descriptor_io io;
  io.save(
    file.path(), nullptr,
    std::make_shared< kv::simple_descriptor_set >( descriptors ) );

  // header + count + type + dimension + 128 floats
  EXPECT_EQ( 7u + 8u + 8u + 1u + 8u + 128u * 4u, file.contents().size() );

  kv::feature_set_sptr feat;
  kv::descriptor_set_sptr read;
  io.load( file.path(), feat, read );

  ASSERT_TRUE( read != nullptr );
  ASSERT_EQ( 1u, read->size() );
  EXPECT_EQ( 128u, read->at( 0 )->size() );
  EXPECT_EQ( 127.0, read->at( 0 )->as_double()[ 127 ] );
}

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}
