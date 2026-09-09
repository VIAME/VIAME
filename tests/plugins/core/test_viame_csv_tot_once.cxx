/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>

#include "read_object_track_set_viame_csv.h"
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
