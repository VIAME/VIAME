/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>

#include <vital/types/bounding_box.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/object_track_set.h>
#include <vital/types/timestamp.h>
#include <vital/util/tokenize.h>

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace kv = kwiver::vital;

// =============================================================================
// Reusable helpers that mirror the process internals, to test the core logic
// independently of the sprokit pipeline infrastructure.
// =============================================================================

struct track_entry
{
  kv::frame_id_t frame_id;
  kv::detected_object_sptr detection;
};

// Column indices for VIAME CSV format
enum
{
  COL_DET_ID = 0,
  COL_SOURCE_ID,
  COL_FRAME_ID,
  COL_MIN_X,
  COL_MIN_Y,
  COL_MAX_X,
  COL_MAX_Y,
  COL_CONFIDENCE,
  COL_LENGTH,
  COL_TOT
};

// ---------------------------------------------------------------------------
// Parse a VIAME CSV file into a per-track map (same logic as process)
static std::map< int, std::vector< track_entry > >
parse_track_csv( const std::string& filename )
{
  std::map< int, std::vector< track_entry > > tracks;

  std::ifstream fin( filename );
  EXPECT_TRUE( fin.good() ) << "Cannot open file: " << filename;

  std::string line;
  while( std::getline( fin, line ) )
  {
    if( line.empty() || line[0] == '#' )
    {
      continue;
    }

    std::vector< std::string > col;
    kv::tokenize( line, col, ",", false );

    if( col.size() < 9 )
    {
      continue;
    }

    int trk_id = atoi( col[COL_DET_ID].c_str() );
    kv::frame_id_t frame_id = atoi( col[COL_FRAME_ID].c_str() );

    kv::bounding_box_d bbox(
      atof( col[COL_MIN_X].c_str() ),
      atof( col[COL_MIN_Y].c_str() ),
      atof( col[COL_MAX_X].c_str() ),
      atof( col[COL_MAX_Y].c_str() ) );

    double conf = atof( col[COL_CONFIDENCE].c_str() );

    kv::detected_object_type_sptr dot =
      std::make_shared< kv::detected_object_type >();

    for( unsigned i = COL_TOT; i + 1 < col.size(); i += 2 )
    {
      if( col[i].empty() || col[i][0] == '(' )
      {
        break;
      }
      dot->set_score( col[i], atof( col[i + 1].c_str() ) );
    }

    kv::detected_object_sptr dob;
    if( COL_TOT < col.size() && !col[COL_TOT].empty() && col[COL_TOT][0] != '(' )
    {
      dob = std::make_shared< kv::detected_object >( bbox, conf, dot );
    }
    else
    {
      dob = std::make_shared< kv::detected_object >( bbox, conf );
    }

    track_entry entry;
    entry.frame_id = frame_id;
    entry.detection = dob;
    tracks[ trk_id ].push_back( entry );
  }

  for( auto& pair : tracks )
  {
    std::sort( pair.second.begin(), pair.second.end(),
      []( const track_entry& a, const track_entry& b )
      {
        return a.frame_id < b.frame_id;
      } );
  }

  return tracks;
}

// ---------------------------------------------------------------------------
// Interpolate between two track entries (same logic as process)
static kv::detected_object_sptr
interpolate( const track_entry& e1, const track_entry& e2,
             kv::frame_id_t target_frame )
{
  double range = static_cast< double >( e2.frame_id - e1.frame_id );
  double alpha = static_cast< double >( target_frame - e1.frame_id ) / range;

  const kv::bounding_box_d& b1 = e1.detection->bounding_box();
  const kv::bounding_box_d& b2 = e2.detection->bounding_box();

  kv::bounding_box_d interp_bbox(
    b1.min_x() * ( 1.0 - alpha ) + b2.min_x() * alpha,
    b1.min_y() * ( 1.0 - alpha ) + b2.min_y() * alpha,
    b1.max_x() * ( 1.0 - alpha ) + b2.max_x() * alpha,
    b1.max_y() * ( 1.0 - alpha ) + b2.max_y() * alpha );

  double conf = e1.detection->confidence();

  kv::detected_object_sptr result;
  if( e1.detection->type() )
  {
    result = std::make_shared< kv::detected_object >(
      interp_bbox, conf, e1.detection->type() );
  }
  else
  {
    result = std::make_shared< kv::detected_object >( interp_bbox, conf );
  }
  return result;
}

// ---------------------------------------------------------------------------
// For a given frame, find or interpolate the detection for a track
static kv::detected_object_sptr
find_or_interpolate( const std::vector< track_entry >& entries,
                     kv::frame_id_t frame_id )
{
  if( entries.empty() )
  {
    return nullptr;
  }

  kv::frame_id_t first_frame = entries.front().frame_id;
  kv::frame_id_t last_frame = entries.back().frame_id;

  if( frame_id < first_frame || frame_id > last_frame )
  {
    return nullptr;
  }

  auto it = std::lower_bound( entries.begin(), entries.end(), frame_id,
    []( const track_entry& e, kv::frame_id_t f )
    {
      return e.frame_id < f;
    } );

  if( it != entries.end() && it->frame_id == frame_id )
  {
    return it->detection;
  }
  else if( it != entries.begin() && it != entries.end() )
  {
    auto prev = std::prev( it );
    return interpolate( *prev, *it, frame_id );
  }

  return nullptr;
}

// ---------------------------------------------------------------------------
// Helper: write a temporary CSV file and return its path
class TempCSVFile
{
public:
  TempCSVFile( const std::string& content )
  {
    char tmpl[] = "/tmp/viame_test_XXXXXX.csv";
    int fd = mkstemps( tmpl, 4 );
    EXPECT_NE( fd, -1 ) << "Failed to create temp file";
    m_path = tmpl;
    close( fd );
    std::ofstream out( m_path );
    out << content;
    out.close();
  }

  ~TempCSVFile()
  {
    std::remove( m_path.c_str() );
  }

  const std::string& path() const { return m_path; }

private:
  std::string m_path;
};

// =============================================================================
// Test: CSV Parsing
// =============================================================================

TEST( resample_object_tracks, csv_parse_basic )
{
  // A simple CSV with 2 tracks, each with 2 states at frames 0 and 10
  std::string csv =
    "# comment line\n"
    "1,video.mp4,0,100,200,300,400,0.9,0,fish,0.8\n"
    "1,video.mp4,10,150,250,350,450,0.85,0,fish,0.75\n"
    "2,video.mp4,0,500,600,700,800,0.7,0,scallop,0.6\n"
    "2,video.mp4,10,510,610,710,810,0.65,0,scallop,0.55\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );

  ASSERT_EQ( tracks.size(), 2u );

  // Track 1
  ASSERT_EQ( tracks[1].size(), 2u );
  EXPECT_EQ( tracks[1][0].frame_id, 0 );
  EXPECT_EQ( tracks[1][1].frame_id, 10 );
  EXPECT_NEAR( tracks[1][0].detection->bounding_box().min_x(), 100.0, 1e-6 );
  EXPECT_NEAR( tracks[1][0].detection->bounding_box().min_y(), 200.0, 1e-6 );
  EXPECT_NEAR( tracks[1][0].detection->bounding_box().max_x(), 300.0, 1e-6 );
  EXPECT_NEAR( tracks[1][0].detection->bounding_box().max_y(), 400.0, 1e-6 );
  EXPECT_NEAR( tracks[1][0].detection->confidence(), 0.9, 1e-6 );

  // Track 2
  ASSERT_EQ( tracks[2].size(), 2u );
  EXPECT_EQ( tracks[2][0].frame_id, 0 );
  EXPECT_NEAR( tracks[2][0].detection->bounding_box().min_x(), 500.0, 1e-6 );
}

// =============================================================================
// Test: CSV with comments and empty lines
// =============================================================================

TEST( resample_object_tracks, csv_parse_comments_and_empty )
{
  std::string csv =
    "# VIAME CSV format\n"
    "\n"
    "# Another comment\n"
    "1,vid.mp4,5,10,20,30,40,0.5,0\n"
    "\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );

  ASSERT_EQ( tracks.size(), 1u );
  ASSERT_EQ( tracks[1].size(), 1u );
  EXPECT_EQ( tracks[1][0].frame_id, 5 );
  EXPECT_NEAR( tracks[1][0].detection->confidence(), 0.5, 1e-6 );
}

// =============================================================================
// Test: CSV with class/score pairs
// =============================================================================

TEST( resample_object_tracks, csv_parse_class_scores )
{
  std::string csv =
    "1,vid.mp4,0,10,20,30,40,0.9,0,fish,0.8,shark,0.1\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );

  ASSERT_EQ( tracks.size(), 1u );
  ASSERT_NE( tracks[1][0].detection->type(), nullptr );

  std::string top_class;
  double top_score;
  tracks[1][0].detection->type()->get_most_likely( top_class, top_score );
  EXPECT_EQ( top_class, "fish" );
  EXPECT_NEAR( top_score, 0.8, 1e-6 );
}

// =============================================================================
// Test: CSV with no class/score pairs
// =============================================================================

TEST( resample_object_tracks, csv_parse_no_classes )
{
  std::string csv =
    "1,vid.mp4,0,10,20,30,40,0.9,0\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );

  ASSERT_EQ( tracks.size(), 1u );
  // Detection should exist, but with no type
  ASSERT_NE( tracks[1][0].detection, nullptr );
  EXPECT_NEAR( tracks[1][0].detection->confidence(), 0.9, 1e-6 );
}

// =============================================================================
// Test: Sorting of unsorted input
// =============================================================================

TEST( resample_object_tracks, csv_parse_unsorted_frames )
{
  std::string csv =
    "1,vid.mp4,20,10,20,30,40,0.9,0\n"
    "1,vid.mp4,0,50,60,70,80,0.8,0\n"
    "1,vid.mp4,10,30,40,50,60,0.85,0\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );

  ASSERT_EQ( tracks[1].size(), 3u );
  EXPECT_EQ( tracks[1][0].frame_id, 0 );
  EXPECT_EQ( tracks[1][1].frame_id, 10 );
  EXPECT_EQ( tracks[1][2].frame_id, 20 );

  // Verify the bbox for frame 0 (was second line in CSV)
  EXPECT_NEAR( tracks[1][0].detection->bounding_box().min_x(), 50.0, 1e-6 );
}

// =============================================================================
// Test: Exact frame match returns original detection
// =============================================================================

TEST( resample_object_tracks, exact_frame_match )
{
  std::string csv =
    "1,vid.mp4,0,100,200,300,400,0.9,0,fish,0.8\n"
    "1,vid.mp4,10,200,300,400,500,0.85,0,fish,0.75\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );
  const auto& entries = tracks[1];

  // Frame 0 - exact match
  auto det = find_or_interpolate( entries, 0 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 100.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().min_y(), 200.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_x(), 300.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_y(), 400.0, 1e-6 );
  EXPECT_NEAR( det->confidence(), 0.9, 1e-6 );

  // Frame 10 - exact match
  det = find_or_interpolate( entries, 10 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 200.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().min_y(), 300.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_x(), 400.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_y(), 500.0, 1e-6 );
  EXPECT_NEAR( det->confidence(), 0.85, 1e-6 );
}

// =============================================================================
// Test: Linear interpolation at midpoint
// =============================================================================

TEST( resample_object_tracks, interpolation_midpoint )
{
  std::string csv =
    "1,vid.mp4,0,100,200,300,400,0.9,0,fish,0.8\n"
    "1,vid.mp4,10,200,300,400,500,0.85,0,fish,0.75\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );
  const auto& entries = tracks[1];

  // Frame 5 - midpoint interpolation (alpha = 0.5)
  auto det = find_or_interpolate( entries, 5 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 150.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().min_y(), 250.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_x(), 350.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_y(), 450.0, 1e-6 );

  // Confidence should be carried forward from earlier state
  EXPECT_NEAR( det->confidence(), 0.9, 1e-6 );
}

// =============================================================================
// Test: Linear interpolation at quarter point
// =============================================================================

TEST( resample_object_tracks, interpolation_quarter )
{
  std::string csv =
    "1,vid.mp4,0,0,0,100,100,0.9,0\n"
    "1,vid.mp4,20,100,100,200,200,0.8,0\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );
  const auto& entries = tracks[1];

  // Frame 5 - quarter interpolation (alpha = 0.25)
  auto det = find_or_interpolate( entries, 5 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 25.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().min_y(), 25.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_x(), 125.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_y(), 125.0, 1e-6 );

  // Frame 15 - three-quarter interpolation (alpha = 0.75)
  det = find_or_interpolate( entries, 15 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 75.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().min_y(), 75.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_x(), 175.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_y(), 175.0, 1e-6 );
}

// =============================================================================
// Test: Interpolation carries forward class labels
// =============================================================================

TEST( resample_object_tracks, interpolation_class_labels_carried_forward )
{
  std::string csv =
    "1,vid.mp4,0,100,200,300,400,0.9,0,fish,0.8,shark,0.1\n"
    "1,vid.mp4,10,200,300,400,500,0.85,0,cod,0.7\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );
  const auto& entries = tracks[1];

  // Interpolated frame should carry labels from earlier state (frame 0)
  auto det = find_or_interpolate( entries, 5 );
  ASSERT_NE( det, nullptr );
  ASSERT_NE( det->type(), nullptr );

  std::string top_class;
  double top_score;
  det->type()->get_most_likely( top_class, top_score );
  EXPECT_EQ( top_class, "fish" );
  EXPECT_NEAR( top_score, 0.8, 1e-6 );
}

// =============================================================================
// Test: Frame outside track range returns null
// =============================================================================

TEST( resample_object_tracks, frame_outside_range )
{
  std::string csv =
    "1,vid.mp4,5,100,200,300,400,0.9,0\n"
    "1,vid.mp4,15,200,300,400,500,0.85,0\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );
  const auto& entries = tracks[1];

  // Before track start
  auto det = find_or_interpolate( entries, 0 );
  EXPECT_EQ( det, nullptr );

  // After track end
  det = find_or_interpolate( entries, 20 );
  EXPECT_EQ( det, nullptr );
}

// =============================================================================
// Test: Multi-segment track interpolation
// =============================================================================

TEST( resample_object_tracks, multi_segment_interpolation )
{
  // Track with 3 keyframes: 0, 10, 20
  std::string csv =
    "1,vid.mp4,0,0,0,100,100,0.9,0\n"
    "1,vid.mp4,10,100,100,200,200,0.8,0\n"
    "1,vid.mp4,20,200,200,300,300,0.7,0\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );
  const auto& entries = tracks[1];

  // Frame 5: between keyframes 0 and 10
  auto det = find_or_interpolate( entries, 5 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 50.0, 1e-6 );
  EXPECT_NEAR( det->confidence(), 0.9, 1e-6 );

  // Frame 15: between keyframes 10 and 20
  det = find_or_interpolate( entries, 15 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 150.0, 1e-6 );
  EXPECT_NEAR( det->confidence(), 0.8, 1e-6 );

  // Frame 10: exact keyframe
  det = find_or_interpolate( entries, 10 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 100.0, 1e-6 );
}

// =============================================================================
// Test: Multiple tracks active on the same frame
// =============================================================================

TEST( resample_object_tracks, multiple_tracks_same_frame )
{
  std::string csv =
    "1,vid.mp4,0,0,0,100,100,0.9,0\n"
    "1,vid.mp4,10,100,100,200,200,0.8,0\n"
    "2,vid.mp4,0,500,500,600,600,0.7,0\n"
    "2,vid.mp4,10,600,600,700,700,0.6,0\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );

  // Frame 5: both tracks should be active
  auto det1 = find_or_interpolate( tracks[1], 5 );
  auto det2 = find_or_interpolate( tracks[2], 5 );

  ASSERT_NE( det1, nullptr );
  ASSERT_NE( det2, nullptr );

  EXPECT_NEAR( det1->bounding_box().min_x(), 50.0, 1e-6 );
  EXPECT_NEAR( det2->bounding_box().min_x(), 550.0, 1e-6 );
}

// =============================================================================
// Test: Track with non-overlapping ranges
// =============================================================================

TEST( resample_object_tracks, non_overlapping_track_ranges )
{
  std::string csv =
    "1,vid.mp4,0,0,0,100,100,0.9,0\n"
    "1,vid.mp4,10,100,100,200,200,0.8,0\n"
    "2,vid.mp4,20,500,500,600,600,0.7,0\n"
    "2,vid.mp4,30,600,600,700,700,0.6,0\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );

  // Frame 5: only track 1 active
  EXPECT_NE( find_or_interpolate( tracks[1], 5 ), nullptr );
  EXPECT_EQ( find_or_interpolate( tracks[2], 5 ), nullptr );

  // Frame 25: only track 2 active
  EXPECT_EQ( find_or_interpolate( tracks[1], 25 ), nullptr );
  EXPECT_NE( find_or_interpolate( tracks[2], 25 ), nullptr );

  // Frame 15: neither track active
  EXPECT_EQ( find_or_interpolate( tracks[1], 15 ), nullptr );
  EXPECT_EQ( find_or_interpolate( tracks[2], 15 ), nullptr );
}

// =============================================================================
// Test: Single-state track
// =============================================================================

TEST( resample_object_tracks, single_state_track )
{
  std::string csv =
    "1,vid.mp4,5,100,200,300,400,0.9,0\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );
  const auto& entries = tracks[1];

  // Exact match
  auto det = find_or_interpolate( entries, 5 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 100.0, 1e-6 );

  // Before and after should return null
  EXPECT_EQ( find_or_interpolate( entries, 4 ), nullptr );
  EXPECT_EQ( find_or_interpolate( entries, 6 ), nullptr );
}

// =============================================================================
// Test: Interpolation with negative coordinates
// =============================================================================

TEST( resample_object_tracks, interpolation_negative_coords )
{
  std::string csv =
    "1,vid.mp4,0,-100,-200,100,200,0.9,0\n"
    "1,vid.mp4,10,100,200,300,400,0.85,0\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );
  const auto& entries = tracks[1];

  // Midpoint
  auto det = find_or_interpolate( entries, 5 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 0.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().min_y(), 0.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_x(), 200.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().max_y(), 300.0, 1e-6 );
}

// =============================================================================
// Test: Interpolation at frame boundaries (alpha=0, alpha=1)
// =============================================================================

TEST( resample_object_tracks, interpolation_boundary_alpha )
{
  std::string csv =
    "1,vid.mp4,0,0,0,100,100,0.9,0\n"
    "1,vid.mp4,10,200,200,300,300,0.8,0\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );
  const auto& entries = tracks[1];

  // Frame 1 (alpha ~= 0.1)
  auto det = find_or_interpolate( entries, 1 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 20.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().min_y(), 20.0, 1e-6 );

  // Frame 9 (alpha ~= 0.9)
  det = find_or_interpolate( entries, 9 );
  ASSERT_NE( det, nullptr );
  EXPECT_NEAR( det->bounding_box().min_x(), 180.0, 1e-6 );
  EXPECT_NEAR( det->bounding_box().min_y(), 180.0, 1e-6 );
}

// =============================================================================
// Test: CSV with attribute columns (parens) stops class parsing
// =============================================================================

TEST( resample_object_tracks, csv_parse_attribute_stop )
{
  std::string csv =
    "1,vid.mp4,0,10,20,30,40,0.9,0,fish,0.8,(poly) 1 2 3 4\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );

  ASSERT_EQ( tracks.size(), 1u );
  ASSERT_NE( tracks[1][0].detection->type(), nullptr );

  std::string top_class;
  double top_score;
  tracks[1][0].detection->type()->get_most_likely( top_class, top_score );
  EXPECT_EQ( top_class, "fish" );
  EXPECT_NEAR( top_score, 0.8, 1e-6 );
}

// =============================================================================
// Test: Empty track set (empty file)
// =============================================================================

TEST( resample_object_tracks, empty_csv )
{
  std::string csv =
    "# Only comments\n"
    "# No data\n";

  TempCSVFile tmp( csv );
  auto tracks = parse_track_csv( tmp.path() );

  EXPECT_TRUE( tracks.empty() );
}
