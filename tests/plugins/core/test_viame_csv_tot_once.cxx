/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>

#include "read_object_track_set_viame_csv.h"
#include "read_detected_object_set_viame_csv.h"
#include "write_object_track_set_viame_csv.h"

#include <vital/types/object_track_set.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace kv = kwiver::vital;

namespace {

using pair_map = std::map< std::string, double >;

std::vector< kv::track_sptr >
make_tracks( int n_states = 3 )
{
  std::vector< kv::track_sptr > tracks;
  const std::vector< std::pair< int, pair_map > > spec = {
    { 1, { { "fish", 0.9 }, { "scallop", 0.1 } } },
    { 2, { { "skate", 0.7 } } } };

  for( auto const& s : spec )
  {
    auto trk = kv::track::create();
    trk->set_id( s.first );
    for( int f = 0; f < n_states; ++f )
    {
      auto dot = std::make_shared< kv::detected_object_type >();
      for( auto const& p : s.second )
      {
        dot->set_score( p.first, p.second );
      }
      auto det = std::make_shared< kv::detected_object >(
        kv::bounding_box_d( 10 + f, 20, 30 + f, 40 ), 0.5, dot );
      trk->append( std::make_shared< kv::object_track_state >( f, f, det ) );
    }
    tracks.push_back( trk );
  }
  return tracks;
}

std::vector< std::string >
write_csv( const fs::path& path, bool once, bool active )
{
  viame::write_object_track_set_viame_csv writer;
  writer.set_write_tot_once( once );
  writer.set_active_writing( active );
  writer.open( path.string() );

  if( active )
  {
    for( int f = 0; f < 3; ++f )
    {
      writer.write_set(
        std::make_shared< kv::object_track_set >( make_tracks( f + 1 ) ),
        kv::timestamp( f, f ), "img" + std::to_string( f ) );
    }
  }
  else
  {
    writer.write_set(
      std::make_shared< kv::object_track_set >( make_tracks() ),
      kv::timestamp( 2, 2 ), "" );
  }
  writer.close();

  std::vector< std::string > rows;
  std::ifstream fin( path );
  for( std::string line; std::getline( fin, line ); )
  {
    if( !line.empty() && line[0] != '#' )
    {
      rows.push_back( line );
    }
  }
  return rows;
}

// track id -> per-state class map
std::map< int, std::vector< pair_map > >
read_csv( const fs::path& path )
{
  viame::read_object_track_set_viame_csv reader;
  reader.set_batch_load( true );
  reader.open( path.string() );

  kv::object_track_set_sptr set;
  EXPECT_TRUE( reader.read_set( set ) );

  std::map< int, std::vector< pair_map > > out;
  for( auto trk : set->tracks() )
  {
    for( auto st : *trk )
    {
      auto* ots = dynamic_cast< kv::object_track_state* >( st.get() );
      pair_map pairs;
      if( ots->detection() && ots->detection()->type() )
      {
        for( auto const& n : ots->detection()->type()->class_names() )
        {
          pairs[ n ] = ots->detection()->type()->score( n );
        }
      }
      out[ trk->id() ].push_back( pairs );
    }
  }
  return out;
}

size_t
count_rows_with( const std::vector< std::string >& rows, const std::string& s )
{
  size_t n = 0;
  for( auto const& r : rows )
  {
    n += ( r.find( s ) != std::string::npos );
  }
  return n;
}

} // namespace

class viame_csv_tmpdir : public ::testing::Test
{
protected:
  void SetUp() override
  {
    m_dir = fs::temp_directory_path() /
            ( "viame_csv_tot_once_" +
              std::to_string( reinterpret_cast< uintptr_t >( this ) ) );
    fs::create_directories( m_dir );
  }

  void TearDown() override
  {
    std::error_code ec;
    fs::remove_all( m_dir, ec );
  }

  fs::path m_dir;
};

class viame_csv_tot_once
  : public viame_csv_tmpdir, public ::testing::WithParamInterface< bool >
{};

TEST_P( viame_csv_tot_once, roundtrip )
{
  const bool active = GetParam();
  const auto full = write_csv( m_dir / "full.csv", false, active );
  const auto once = write_csv( m_dir / "once.csv", true, active );

  ASSERT_EQ( full.size(), 6u );
  ASSERT_EQ( once.size(), 6u );

  EXPECT_EQ( count_rows_with( full, ",fish," ), 3u );
  EXPECT_EQ( count_rows_with( full, ",skate," ), 3u );
  EXPECT_EQ( count_rows_with( once, ",fish," ), 1u );
  EXPECT_EQ( count_rows_with( once, ",scallop," ), 1u );
  EXPECT_EQ( count_rows_with( once, ",skate," ), 1u );

  // The first row of each track keeps its pairs
  EXPECT_NE( once[0].find( ",fish,0.9,scallop,0.1" ), std::string::npos );

  const auto expected = read_csv( m_dir / "full.csv" );
  ASSERT_EQ( expected.size(), 2u );
  ASSERT_EQ( expected.at( 1 ).size(), 3u );

  EXPECT_EQ( read_csv( m_dir / "once.csv" ), expected );
}

TEST_F( viame_csv_tmpdir, reader_fills_from_first_or_last_row )
{
  const auto path = m_dir / "sparse.csv";
  {
    std::ofstream out( path );
    out << "1,img0,0,10,20,30,40,0.5,0,fish,0.9,scallop,0.1\n"
        << "1,img1,1,11,20,31,40,0.5,0\n"
        << "1,img2,2,12,20,32,40,0.5,0,(poly) 12 20 32 20 32 40\n"
        << "2,img0,0,10,20,30,40,0.5,0\n"
        << "2,img1,1,11,20,31,40,0.5,0\n"
        << "2,img2,2,12,20,32,40,0.5,0,skate,0.7\n"
        << "3,img0,0,10,20,30,40,0.5,0,fish,0.9\n"
        << "3,img1,1,11,20,31,40,0.5,0,skate,0.8\n"
        << "3,img2,2,12,20,32,40,0.5,0\n"
        << "4,img0,0,10,20,30,40,0.5,0\n"
        << "4,img1,1,11,20,31,40,0.5,0\n";
  }

  const auto got = read_csv( path );
  const pair_map fish = { { "fish", 0.9 }, { "scallop", 0.1 } };
  const pair_map skate = { { "skate", 0.7 } };

  EXPECT_EQ( got.at( 1 ), ( std::vector< pair_map >{ fish, fish, fish } ) );
  EXPECT_EQ( got.at( 2 ), ( std::vector< pair_map >{ skate, skate, skate } ) );
  EXPECT_EQ( got.at( 3 )[2], ( pair_map{ { "skate", 0.8 } } ) );
  EXPECT_TRUE( got.at( 4 )[0].empty() );
  EXPECT_TRUE( got.at( 4 )[1].empty() );
}

INSTANTIATE_TEST_CASE_P( modes, viame_csv_tot_once, ::testing::Bool() );

TEST_F( viame_csv_tmpdir, polygons_preserve_other_optional_fields )
{
  const auto path = m_dir / "mixed_fields.csv";
  {
    std::ofstream out( path );
    out << "1,image.png,0,10,20,30,40,0.5,12.5,fish,0.9,"
        << "(atr) source image001.png,(poly) 10 20 15 20 15 25,"
        << "(kp) head 12 22,(atr) score 7.5,(+poly) 25 35 30 35 30 40,"
        << "(note) reviewed fish,(atr) verified\n";
  }
  const auto check = []( kv::detected_object_sptr const& det ) {
    EXPECT_EQ( det->get_flattened_polygons().size(), 2 );
    EXPECT_DOUBLE_EQ( det->get_attribute< double >( "length" ), 12.5 );
    EXPECT_DOUBLE_EQ( det->type()->score( "fish" ), 0.9 );
    EXPECT_EQ( det->notes(), ( std::vector< std::string >{
      ":source=image001.png", ":score=7.5", "reviewed fish", ":verified=true" } ) );
    ASSERT_EQ( det->keypoints().count( "head" ), 1 );
    EXPECT_DOUBLE_EQ( det->keypoints().at( "head" ).value()[0], 12 );
    EXPECT_DOUBLE_EQ( det->keypoints().at( "head" ).value()[1], 22 );
  };
  viame::read_detected_object_set_viame_csv detections;
  detections.open( path.string() );
  kv::detected_object_set_sptr set;
  std::string image_name;
  ASSERT_TRUE( detections.read_set( set, image_name ) );
  ASSERT_EQ( set->size(), 1 );
  check( *set->begin() );
  detections.close();

  viame::read_object_track_set_viame_csv tracks;
  tracks.set_batch_load( true );
  tracks.open( path.string() );
  kv::object_track_set_sptr track_set;
  ASSERT_TRUE( tracks.read_set( track_set ) );
  ASSERT_EQ( track_set->tracks().size(), 1 );
  auto state = std::dynamic_pointer_cast< kv::object_track_state >(
    *track_set->tracks()[0]->begin() );
  ASSERT_TRUE( state );
  check( state->detection() );
  tracks.close();
}

TEST( viame_csv, multiple_polygon_pieces_roundtrip )
{
  const auto path = fs::temp_directory_path() / "viame_csv_multipolygon_tracks.csv";
  auto tracks = make_tracks( 1 );
  const std::vector< std::vector< double > > polygons = {
    { 10, 20, 15, 20, 15, 25, 10, 25 },
    { 25, 35, 30, 35, 30, 40, 25, 40 } };
  for( auto const& track : tracks )
  {
    auto state = std::dynamic_pointer_cast< kv::object_track_state >( *track->begin() );
    state->detection()->set_flattened_polygons( polygons );
  }
  viame::write_object_track_set_viame_csv writer;
  writer.open( path.string() );
  writer.write_set( std::make_shared< kv::object_track_set >( tracks ),
                    kv::timestamp( 0, 0 ), "image.png" );
  writer.close();
  viame::read_object_track_set_viame_csv reader;
  reader.set_batch_load( true );
  reader.open( path.string() );
  kv::object_track_set_sptr result;
  ASSERT_TRUE( reader.read_set( result ) );
  ASSERT_EQ( tracks.size(), result->tracks().size() );
  for( auto const& track : result->tracks() )
  {
    auto state = std::dynamic_pointer_cast< kv::object_track_state >( *track->begin() );
    EXPECT_EQ( polygons, state->detection()->get_flattened_polygons() );
  }
  reader.close();
  fs::remove( path );
}

#include "read_detected_object_set_dive.h"
#include "write_object_track_set_dive.h"
#include <sstream>

TEST( viame_csv_tot_once, centerline_dive_roundtrip )
{
  auto tracks = make_tracks( 1 );
  auto state = std::dynamic_pointer_cast< kv::object_track_state >( *tracks[0]->begin() );
  auto det = state->detection();
  det->add_keypoint( "head", { 10.123456789, 20 } );
  det->add_keypoint( "spine_010", { 28, 28 } );
  det->add_keypoint( "spine_002", { 22, 30.123456789 } );
  det->add_keypoint( "tail", { 30, 20 } );
  det->add_keypoint( "eye", { 12, 21 } );
  std::ostringstream json;
  viame::write_dive_json( json, tracks, {} );
  viame::dive_annotation_file parsed;
  ASSERT_TRUE( viame::parse_dive_json_manual( json.str(), kv::get_logger( "test" ), parsed ) );
  const auto& feature = parsed.tracks.at( "1" ).features[0];
  ASSERT_EQ( feature.centerline.size(), 4 );
  EXPECT_DOUBLE_EQ( feature.centerline[1][1], 30.123456789 );
  auto restored = viame::create_detected_object_from_dive( feature, {} );
  ASSERT_EQ( restored->keypoints().size(), 5 );
  EXPECT_DOUBLE_EQ( restored->keypoints().at( "head" ).value()[0], 10.123456789 );
  EXPECT_DOUBLE_EQ( restored->keypoints().at( "spine_001" ).value()[1], 30.123456789 );
  EXPECT_TRUE( restored->keypoints().count( "eye" ) );
  const auto csv_path = fs::temp_directory_path() / "viame_centerline_roundtrip.csv";
  viame::write_object_track_set_viame_csv writer;
  writer.set_active_writing( false );
  writer.open( csv_path.string() );
  writer.write_set( std::make_shared< kv::object_track_set >( tracks ), kv::timestamp( 0, 0 ), "img.png" );
  writer.close();
  viame::read_object_track_set_viame_csv reader;
  reader.set_batch_load( true );
  reader.open( csv_path.string() );
  kv::object_track_set_sptr loaded;
  ASSERT_TRUE( reader.read_set( loaded ) );
  auto first = std::dynamic_pointer_cast< kv::object_track_state >( *loaded->tracks()[0]->begin() );
  EXPECT_DOUBLE_EQ( first->detection()->keypoints().at( "spine_002" ).value()[1], 30.123456789 );
  EXPECT_DOUBLE_EQ( first->detection()->keypoints().at( "head" ).value()[0], 10.123456789 );
  fs::remove( csv_path );
}
