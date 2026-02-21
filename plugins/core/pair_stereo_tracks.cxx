/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Shared stereo track pairing implementation
 */

#include "pair_stereo_tracks.h"
#include "utilities_target_clfr.h"

#include <algorithm>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace viame
{

namespace core
{

// =============================================================================
// track_union_find
// =============================================================================

kv::track_id_t
track_union_find
::find( kv::track_id_t x )
{
  if( parent.find( x ) == parent.end() )
  {
    parent[x] = x;
    rnk[x] = 0;
  }
  if( parent[x] != x )
  {
    parent[x] = find( parent[x] );
  }
  return parent[x];
}

void
track_union_find
::unite( kv::track_id_t a, kv::track_id_t b )
{
  kv::track_id_t ra = find( a );
  kv::track_id_t rb = find( b );
  if( ra == rb )
    return;
  if( rnk[ra] < rnk[rb] )
    std::swap( ra, rb );
  parent[rb] = ra;
  if( rnk[ra] == rnk[rb] )
    rnk[ra]++;
}

std::map< kv::track_id_t, std::set< kv::track_id_t > >
track_union_find
::groups()
{
  std::map< kv::track_id_t, std::set< kv::track_id_t > > result;
  for( const auto& p : parent )
  {
    result[find( p.first )].insert( p.first );
  }
  return result;
}

// =============================================================================
// Free helper functions
// =============================================================================

kv::detected_object_type_sptr
compute_stereo_average_classification(
  const std::vector< kv::detected_object_sptr >& dets_left,
  const std::vector< kv::detected_object_sptr >& dets_right,
  bool weighted,
  bool scale_by_conf,
  const std::string& ignore_class )
{
  // Concatenate into a single vector and delegate to the shared utility
  std::vector< kv::detected_object_sptr > all_dets;
  all_dets.reserve( dets_left.size() + dets_right.size() );
  all_dets.insert( all_dets.end(), dets_left.begin(), dets_left.end() );
  all_dets.insert( all_dets.end(), dets_right.begin(), dets_right.end() );

  return compute_average_classification( all_dets, weighted,
                                         scale_by_conf, ignore_class );
}

// -----------------------------------------------------------------------------
void
apply_classification_to_track(
  const kv::track_sptr& trk,
  const kv::detected_object_type_sptr& dot )
{
  if( !trk || !dot )
    return;

  for( auto it = trk->begin(); it != trk->end(); ++it )
  {
    auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
    if( ots && ots->detection() )
    {
      ots->detection()->set_type( dot );
    }
  }
}

// =============================================================================
// stereo_track_pairer
// =============================================================================

stereo_track_pairer
::stereo_track_pairer()
{
}

stereo_track_pairer
::~stereo_track_pairer()
{
}

// -----------------------------------------------------------------------------
kv::config_block_sptr
stereo_track_pairer
::get_configuration() const
{
  auto config = kv::config_block::empty_config();

  config->set_value( "accumulate_track_pairings",
    std::to_string( m_accumulate_track_pairings ),
    "If true, accumulate track pairings across frames and resolve at stream end. "
    "Requires track inputs (object_track_set1/2). When false, operates per-frame." );

  config->set_value( "pairing_resolution_method",
    m_pairing_resolution_method,
    "How to resolve accumulated track pairings. 'most_likely' picks the right track "
    "with most frame co-occurrences for each left track. 'split' creates separate "
    "tracks for each consistent pairing segment." );

  config->set_value( "detection_split_threshold",
    std::to_string( m_detection_split_threshold ),
    "Minimum number of frame pairings required to keep a split segment. "
    "Used with pairing_resolution_method='split'." );

  config->set_value( "min_track_length",
    std::to_string( m_min_track_length ),
    "Minimum number of detections per output track. Tracks shorter than this "
    "are filtered out. 0 disables this filter." );

  config->set_value( "max_track_length",
    std::to_string( m_max_track_length ),
    "Maximum number of detections per output track. Tracks longer than this "
    "are filtered out. 0 disables this filter." );

  config->set_value( "min_avg_surface_area",
    std::to_string( m_min_avg_surface_area ),
    "Minimum average bounding box area (in pixels) across track detections. "
    "Tracks with smaller average area are filtered out. 0 disables this filter." );

  config->set_value( "max_avg_surface_area",
    std::to_string( m_max_avg_surface_area ),
    "Maximum average bounding box area (in pixels) across track detections. "
    "Tracks with larger average area are filtered out. 0 disables this filter." );

  config->set_value( "average_stereo_classes",
    std::to_string( m_average_stereo_classes ),
    "If true, average class labels across matched stereo track pairs so both "
    "cameras report the same species classification." );

  config->set_value( "class_averaging_method",
    m_class_averaging_method,
    "Method for averaging class labels: 'weighted_average' (weight by detection "
    "confidence), 'simple_average' (equal weight per detection), or "
    "'weighted_scaled_by_conf' (weighted average scaled by 0.1+0.9*avg_conf)." );

  config->set_value( "class_averaging_ignore_class",
    m_class_averaging_ignore_class,
    "Optional class name to exclude from averaging when mixed with real classes. "
    "Detections whose only label is this class are separated; if non-ignored "
    "detections are also present the ignored class is excluded from the output. "
    "Empty string disables this feature." );

  config->set_value( "output_unmatched",
    std::to_string( m_output_unmatched ),
    "If true, output unmatched detections as separate tracks with unique IDs. "
    "If false, only output matched detection pairs." );

  return config;
}

// -----------------------------------------------------------------------------
void
stereo_track_pairer
::set_configuration( kv::config_block_sptr config )
{
  m_accumulate_track_pairings =
    config->get_value< bool >( "accumulate_track_pairings", m_accumulate_track_pairings );
  m_pairing_resolution_method =
    config->get_value< std::string >( "pairing_resolution_method", m_pairing_resolution_method );
  m_detection_split_threshold =
    config->get_value< int >( "detection_split_threshold", m_detection_split_threshold );
  m_min_track_length =
    config->get_value< int >( "min_track_length", m_min_track_length );
  m_max_track_length =
    config->get_value< int >( "max_track_length", m_max_track_length );
  m_min_avg_surface_area =
    config->get_value< double >( "min_avg_surface_area", m_min_avg_surface_area );
  m_max_avg_surface_area =
    config->get_value< double >( "max_avg_surface_area", m_max_avg_surface_area );
  m_average_stereo_classes =
    config->get_value< bool >( "average_stereo_classes", m_average_stereo_classes );
  m_class_averaging_method =
    config->get_value< std::string >( "class_averaging_method", m_class_averaging_method );
  m_class_averaging_ignore_class =
    config->get_value< std::string >( "class_averaging_ignore_class", m_class_averaging_ignore_class );
  m_output_unmatched =
    config->get_value< bool >( "output_unmatched", m_output_unmatched );

  // Validate class averaging method
  if( m_average_stereo_classes )
  {
    if( m_class_averaging_method != "weighted_average" &&
        m_class_averaging_method != "simple_average" &&
        m_class_averaging_method != "weighted_scaled_by_conf" )
    {
      throw std::runtime_error( "Invalid class_averaging_method: '" +
                                m_class_averaging_method +
                                "'. Must be 'weighted_average', 'simple_average', or "
                                "'weighted_scaled_by_conf'." );
    }
  }

  // Validate pairing resolution method when accumulation is enabled
  if( m_accumulate_track_pairings )
  {
    if( m_pairing_resolution_method != "most_likely" &&
        m_pairing_resolution_method != "split" )
    {
      throw std::runtime_error( "Invalid pairing_resolution_method: '" +
                                m_pairing_resolution_method +
                                "'. Must be 'most_likely' or 'split'." );
    }
  }
}

// -----------------------------------------------------------------------------
bool
stereo_track_pairer
::accumulation_enabled() const
{
  return m_accumulate_track_pairings;
}

// -----------------------------------------------------------------------------
bool
stereo_track_pairer
::output_unmatched() const
{
  return m_output_unmatched;
}

// -----------------------------------------------------------------------------
bool
stereo_track_pairer
::average_stereo_classes() const
{
  return m_average_stereo_classes;
}

// -----------------------------------------------------------------------------
bool
stereo_track_pairer
::use_weighted_averaging() const
{
  return m_class_averaging_method == "weighted_average" ||
         m_class_averaging_method == "weighted_scaled_by_conf";
}

// -----------------------------------------------------------------------------
bool
stereo_track_pairer
::use_scaled_by_conf() const
{
  return m_class_averaging_method == "weighted_scaled_by_conf";
}

// -----------------------------------------------------------------------------
std::string
stereo_track_pairer
::class_averaging_ignore_class() const
{
  return m_class_averaging_ignore_class;
}

// -----------------------------------------------------------------------------
kv::track_id_t
stereo_track_pairer
::allocate_track_id()
{
  return m_next_track_id++;
}

// -----------------------------------------------------------------------------
void
stereo_track_pairer
::remap_tracks_per_frame(
  const kv::object_track_set_sptr& tracks1,
  const kv::object_track_set_sptr& tracks2,
  const std::vector< std::pair< int, int > >& matches,
  const std::vector< kv::track_id_t >& track_ids1,
  const std::vector< kv::track_id_t >& track_ids2,
  std::vector< kv::track_sptr >& output1,
  std::vector< kv::track_sptr >& output2 )
{
  // Register matches in union-find with namespace encoding:
  //   Left IDs stored as-is; Right IDs as -(right_id + 1)
  for( const auto& match : matches )
  {
    kv::track_id_t left_id = track_ids1[match.first];
    kv::track_id_t right_id = track_ids2[match.second];
    m_track_union_find.unite( left_id, -( right_id + 1 ) );
  }

  // Get connected components and assign output IDs
  auto groups = m_track_union_find.groups();
  for( const auto& group : groups )
  {
    // Check if any member already has an output ID
    kv::track_id_t output_id = -1;
    for( kv::track_id_t member : group.second )
    {
      if( member >= 0 )
      {
        auto it = m_left_to_output_id.find( member );
        if( it != m_left_to_output_id.end() )
        {
          output_id = it->second;
          break;
        }
      }
      else
      {
        kv::track_id_t orig_right = -( member + 1 );
        auto it = m_right_to_output_id.find( orig_right );
        if( it != m_right_to_output_id.end() )
        {
          output_id = it->second;
          break;
        }
      }
    }

    // Allocate new ID if needed
    if( output_id < 0 )
    {
      output_id = m_next_track_id++;
    }

    // Apply output ID to all members in this connected component
    for( kv::track_id_t member : group.second )
    {
      if( member >= 0 )
      {
        m_left_to_output_id[member] = output_id;
      }
      else
      {
        kv::track_id_t orig_right = -( member + 1 );
        m_right_to_output_id[orig_right] = output_id;
      }
    }
  }

  // Build remapped left tracks — always output all tracks in pass-through
  // mode (upstream tracker tracks should never be dropped).  Use a map so
  // that transitively linked tracks (same output ID) get merged into one
  // track object instead of creating duplicates that downstream code drops.
  std::map< kv::track_id_t, kv::track_sptr > left_output_map;
  for( const auto& trk : tracks1->tracks() )
  {
    kv::track_id_t output_id;
    auto map_it = m_left_to_output_id.find( trk->id() );

    if( map_it != m_left_to_output_id.end() )
    {
      output_id = map_it->second;
    }
    else
    {
      output_id = m_next_track_id++;
      m_left_to_output_id[trk->id()] = output_id;
    }

    if( left_output_map.find( output_id ) == left_output_map.end() )
    {
      auto new_trk = kv::track::create();
      new_trk->set_id( output_id );
      left_output_map[output_id] = new_trk;
    }

    for( auto it = trk->begin(); it != trk->end(); ++it )
    {
      auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
      if( ots )
      {
        auto new_state = std::make_shared< kv::object_track_state >(
          ots->frame(), ots->time(), ots->detection() );
        left_output_map[output_id]->append( new_state );
      }
    }
  }

  std::vector< kv::track_sptr > remapped_left;
  for( auto& entry : left_output_map )
    remapped_left.push_back( entry.second );

  // Build remapped right tracks — same merge-by-output-ID approach
  std::map< kv::track_id_t, kv::track_sptr > right_output_map;
  for( const auto& trk : tracks2->tracks() )
  {
    kv::track_id_t output_id;
    auto map_it = m_right_to_output_id.find( trk->id() );

    if( map_it != m_right_to_output_id.end() )
    {
      output_id = map_it->second;
    }
    else
    {
      output_id = m_next_track_id++;
      m_right_to_output_id[trk->id()] = output_id;
    }

    if( right_output_map.find( output_id ) == right_output_map.end() )
    {
      auto new_trk = kv::track::create();
      new_trk->set_id( output_id );
      right_output_map[output_id] = new_trk;
    }

    for( auto it = trk->begin(); it != trk->end(); ++it )
    {
      auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
      if( ots )
      {
        auto new_state = std::make_shared< kv::object_track_state >(
          ots->frame(), ots->time(), ots->detection() );
        right_output_map[output_id]->append( new_state );
      }
    }
  }

  std::vector< kv::track_sptr > remapped_right;
  for( auto& entry : right_output_map )
    remapped_right.push_back( entry.second );

  // Apply class averaging across matched stereo pairs if enabled
  if( m_average_stereo_classes )
  {
    std::map< kv::track_id_t, kv::track_sptr > left_by_id, right_by_id;
    for( const auto& trk : remapped_left )
      left_by_id[trk->id()] = trk;
    for( const auto& trk : remapped_right )
      right_by_id[trk->id()] = trk;

    bool weighted = use_weighted_averaging();
    bool sbc = use_scaled_by_conf();
    for( auto& entry : left_by_id )
    {
      auto rit = right_by_id.find( entry.first );
      if( rit == right_by_id.end() )
        continue;

      std::vector< kv::detected_object_sptr > left_dets, right_dets;
      for( auto it = entry.second->begin(); it != entry.second->end(); ++it )
      {
        auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
        if( ots && ots->detection() )
          left_dets.push_back( ots->detection() );
      }
      for( auto it = rit->second->begin(); it != rit->second->end(); ++it )
      {
        auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
        if( ots && ots->detection() )
          right_dets.push_back( ots->detection() );
      }

      auto avg_dot = compute_stereo_average_classification(
        left_dets, right_dets, weighted, sbc, m_class_averaging_ignore_class );
      if( avg_dot )
      {
        apply_classification_to_track( entry.second, avg_dot );
        apply_classification_to_track( rit->second, avg_dot );
      }
    }
  }

  output1 = remapped_left;
  output2 = remapped_right;
}

// =============================================================================
// Accumulation method implementations
// =============================================================================

size_t
stereo_track_pairer
::cantor_pairing_fn( size_t i, size_t j )
{
  return ( ( i + j ) * ( i + j + 1u ) ) / 2u + j;
}

// -----------------------------------------------------------------------------
void
stereo_track_pairer
::accumulate_frame_pairings(
  const std::vector< std::pair< int, int > >& matches,
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const std::vector< kv::track_id_t >& track_ids1,
  const std::vector< kv::track_id_t >& track_ids2,
  const kv::timestamp& timestamp )
{
  // Store all detections into accumulated track maps
  for( size_t i = 0; i < detections1.size(); ++i )
  {
    kv::track_id_t tid = track_ids1[i];
    if( m_accumulated_tracks1.find( tid ) == m_accumulated_tracks1.end() )
    {
      m_accumulated_tracks1[tid] = kv::track::create();
      m_accumulated_tracks1[tid]->set_id( tid );
    }
    auto state = std::make_shared< kv::object_track_state >( timestamp, detections1[i] );
    m_accumulated_tracks1[tid]->append( state );
  }

  for( size_t i = 0; i < detections2.size(); ++i )
  {
    kv::track_id_t tid = track_ids2[i];
    if( m_accumulated_tracks2.find( tid ) == m_accumulated_tracks2.end() )
    {
      m_accumulated_tracks2[tid] = kv::track::create();
      m_accumulated_tracks2[tid]->set_id( tid );
    }
    auto state = std::make_shared< kv::object_track_state >( timestamp, detections2[i] );
    m_accumulated_tracks2[tid]->append( state );
  }

  // Record pairings using cantor pairing key
  for( const auto& match : matches )
  {
    kv::track_id_t left_id = track_ids1[match.first];
    kv::track_id_t right_id = track_ids2[match.second];
    size_t key = cantor_pairing_fn( static_cast< size_t >( left_id ),
                                    static_cast< size_t >( right_id ) );

    if( m_left_to_right_pairing.find( key ) == m_left_to_right_pairing.end() )
    {
      m_left_to_right_pairing[key] = pairing{ {}, { left_id, right_id } };
    }

    m_left_to_right_pairing[key].frame_set.insert( timestamp.get_frame() );
  }
}

// -----------------------------------------------------------------------------
kv::track_id_t
stereo_track_pairer
::last_accumulated_track_id() const
{
  kv::track_id_t max_id = 0;

  for( const auto& pair : m_accumulated_tracks1 )
  {
    if( pair.second->id() > max_id )
      max_id = pair.second->id();
  }
  for( const auto& pair : m_accumulated_tracks2 )
  {
    if( pair.second->id() > max_id )
      max_id = pair.second->id();
  }

  return max_id;
}

// -----------------------------------------------------------------------------
void
stereo_track_pairer
::select_most_likely_pairing(
  std::vector< kv::track_sptr >& left_tracks,
  std::vector< kv::track_sptr >& right_tracks,
  std::set< kv::track_id_t >& proc_left,
  std::set< kv::track_id_t >& proc_right )
{
  // Build union-find from all accumulated pairings for transitive association.
  // Left IDs stored as-is; Right IDs encoded as -(right_id + 1).
  track_union_find uf;
  for( const auto& pair : m_left_to_right_pairing )
  {
    kv::track_id_t left_id = pair.second.left_right_id_pair.left_id;
    kv::track_id_t right_id = pair.second.left_right_id_pair.right_id;
    uf.unite( left_id, -( right_id + 1 ) );
  }

  auto groups = uf.groups();
  kv::track_id_t next_id = last_accumulated_track_id() + 1;

  for( const auto& group : groups )
  {
    // Separate left and right members
    std::set< kv::track_id_t > left_members, right_members;
    for( kv::track_id_t member : group.second )
    {
      if( member >= 0 )
        left_members.insert( member );
      else
        right_members.insert( -( member + 1 ) );
    }

    // Only process components that have both left and right members
    if( left_members.empty() || right_members.empty() )
      continue;

    kv::track_id_t output_id = next_id++;

    // Merge all left member tracks into one output track
    auto merged_left = kv::track::create();
    merged_left->set_id( output_id );
    for( kv::track_id_t lid : left_members )
    {
      if( m_accumulated_tracks1.find( lid ) == m_accumulated_tracks1.end() )
        continue;
      for( auto it = m_accumulated_tracks1[lid]->begin();
           it != m_accumulated_tracks1[lid]->end(); ++it )
      {
        auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
        if( ots )
        {
          auto new_state = std::make_shared< kv::object_track_state >(
            ots->frame(), ots->time(), ots->detection() );
          merged_left->append( new_state );
        }
      }
      proc_left.insert( lid );
    }

    // Merge all right member tracks into one output track
    auto merged_right = kv::track::create();
    merged_right->set_id( output_id );
    for( kv::track_id_t rid : right_members )
    {
      if( m_accumulated_tracks2.find( rid ) == m_accumulated_tracks2.end() )
        continue;
      for( auto it = m_accumulated_tracks2[rid]->begin();
           it != m_accumulated_tracks2[rid]->end(); ++it )
      {
        auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
        if( ots )
        {
          auto new_state = std::make_shared< kv::object_track_state >(
            ots->frame(), ots->time(), ots->detection() );
          merged_right->append( new_state );
        }
      }
      proc_right.insert( rid );
    }

    if( merged_left->size() > 0 )
      left_tracks.push_back( merged_left );
    if( merged_right->size() > 0 )
      right_tracks.push_back( merged_right );
  }
}

// -----------------------------------------------------------------------------
std::vector< split_range >
stereo_track_pairer
::create_split_ranges_from_track_pairs() const
{
  // Find last pairing frame id
  kv::frame_id_t last_frame_id = 0;
  for( const auto& p : m_left_to_right_pairing )
  {
    if( !p.second.frame_set.empty() )
    {
      kv::frame_id_t last = *p.second.frame_set.rbegin();
      if( last > last_frame_id )
        last_frame_id = last;
    }
  }

  kv::track_id_t next_id = last_accumulated_track_id() + 1;

  // Track open and pending ranges
  std::map< size_t, std::shared_ptr< split_range > > open_ranges;
  std::set< std::shared_ptr< split_range > > pending_ranges;
  std::vector< split_range > ranges;

  for( kv::frame_id_t i_frame = 0; i_frame <= last_frame_id; i_frame++ )
  {
    for( const auto& p : m_left_to_right_pairing )
    {
      // Skip if this pairing is not in current frame
      if( p.second.frame_set.find( i_frame ) == p.second.frame_set.end() )
        continue;

      if( open_ranges.find( p.first ) != open_ranges.end() )
      {
        // Update existing open range
        auto& range = open_ranges[p.first];
        range->detection_count += 1;
        range->frame_id_last = i_frame + 1;

        // Remove pending ranges that conflict with this now-confirmed range
        if( range->detection_count >= m_detection_split_threshold )
        {
          std::set< std::shared_ptr< split_range > > to_remove;
          for( const auto& pending : pending_ranges )
          {
            if( pending == range )
              continue;
            if( pending->left_id == range->left_id || pending->right_id == range->right_id )
              to_remove.insert( pending );
          }

          // Remove source from pending if it passed threshold
          if( pending_ranges.find( range ) != pending_ranges.end() )
            pending_ranges.erase( range );

          // Move removed ranges to processed if they meet threshold
          for( const auto& r : to_remove )
          {
            pending_ranges.erase( r );
            if( r->detection_count >= m_detection_split_threshold )
            {
              r->frame_id_last = range->frame_id_first - 1;
              ranges.push_back( *r );
            }
          }

          // Remove from open_ranges map
          for( auto it = open_ranges.begin(); it != open_ranges.end(); )
          {
            if( to_remove.find( it->second ) != to_remove.end() )
              it = open_ranges.erase( it );
            else
              ++it;
          }
        }
      }
      else
      {
        // Create new range
        auto range = std::make_shared< split_range >();
        range->left_id = p.second.left_right_id_pair.left_id;
        range->right_id = p.second.left_right_id_pair.right_id;
        range->new_track_id = next_id++;
        range->detection_count = 1;
        range->frame_id_first = i_frame;
        range->frame_id_last = i_frame + 1;

        open_ranges[p.first] = range;
        pending_ranges.insert( range );

        // Mark overlapping open ranges as pending
        for( auto& op : open_ranges )
        {
          if( op.second == range )
            continue;
          if( op.second->left_id == range->left_id || op.second->right_id == range->right_id )
            pending_ranges.insert( op.second );
        }
      }
    }
  }

  // Finalize remaining open ranges
  for( auto& op : open_ranges )
  {
    if( op.second->detection_count < m_detection_split_threshold )
      continue;

    op.second->frame_id_last = std::numeric_limits< int64_t >::max();
    ranges.push_back( *op.second );
  }

  return ranges;
}

// -----------------------------------------------------------------------------
void
stereo_track_pairer
::split_paired_tracks(
  std::vector< kv::track_sptr >& left_tracks,
  std::vector< kv::track_sptr >& right_tracks,
  std::set< kv::track_id_t >& proc_left,
  std::set< kv::track_id_t >& proc_right )
{
  auto ranges = create_split_ranges_from_track_pairs();

  for( const auto& range : ranges )
  {
    proc_left.insert( range.left_id );
    proc_right.insert( range.right_id );

    // Create new tracks from frame ranges
    auto split_track = []( const kv::track_sptr& source, const split_range& r )
    {
      auto new_track = kv::track::create();
      new_track->set_id( r.new_track_id );

      for( const auto& state : *source | kv::as_object_track )
      {
        if( state->frame() >= r.frame_id_first && state->frame() < r.frame_id_last )
        {
          new_track->append( state );
        }
      }

      return new_track;
    };

    left_tracks.push_back( split_track( m_accumulated_tracks1[range.left_id], range ) );
    right_tracks.push_back( split_track( m_accumulated_tracks2[range.right_id], range ) );
  }
}

// -----------------------------------------------------------------------------
std::vector< kv::track_sptr >
stereo_track_pairer
::filter_tracks( std::vector< kv::track_sptr > tracks ) const
{
  bool has_length_filter = ( m_min_track_length > 0 ) || ( m_max_track_length > 0 );
  bool has_area_filter = ( m_min_avg_surface_area > 0.0 ) || ( m_max_avg_surface_area > 0.0 );

  if( !has_length_filter && !has_area_filter )
    return tracks;

  if( has_length_filter )
  {
    int min_len = m_min_track_length > 0 ? m_min_track_length : 0;
    int max_len = m_max_track_length > 0 ? m_max_track_length : std::numeric_limits< int >::max();

    tracks.erase(
      std::remove_if( tracks.begin(), tracks.end(),
        [min_len, max_len]( const kv::track_sptr& track )
        {
          int sz = static_cast< int >( track->size() );
          return sz < min_len || sz > max_len;
        } ),
      tracks.end() );
  }

  if( has_area_filter )
  {
    double min_area = m_min_avg_surface_area > 0.0 ? m_min_avg_surface_area : 0.0;
    double max_area = m_max_avg_surface_area > 0.0
      ? m_max_avg_surface_area : std::numeric_limits< double >::max();

    tracks.erase(
      std::remove_if( tracks.begin(), tracks.end(),
        [min_area, max_area]( const kv::track_sptr& track )
        {
          double avg_area = 0.0;
          int count = 0;
          for( const auto& state : *track | kv::as_object_track )
          {
            if( state->detection() )
            {
              avg_area += state->detection()->bounding_box().area();
              count++;
            }
          }
          if( count > 0 )
            avg_area /= static_cast< double >( count );

          return avg_area < min_area || avg_area > max_area;
        } ),
      tracks.end() );
  }

  return tracks;
}

// -----------------------------------------------------------------------------
void
stereo_track_pairer
::resolve_accumulated_pairings(
  std::vector< kv::track_sptr >& output_trks1,
  std::vector< kv::track_sptr >& output_trks2 )
{
  std::set< kv::track_id_t > proc_left, proc_right;

  // Call resolution method
  if( m_pairing_resolution_method == "most_likely" )
  {
    select_most_likely_pairing( output_trks1, output_trks2, proc_left, proc_right );
  }
  else // "split"
  {
    split_paired_tracks( output_trks1, output_trks2, proc_left, proc_right );
  }

  // Append unmatched tracks with sequential IDs (no gaps)
  if( m_output_unmatched )
  {
    // Find max existing output ID
    kv::track_id_t max_output_id = 0;
    for( const auto& trk : output_trks1 )
    {
      if( trk->id() > max_output_id )
        max_output_id = trk->id();
    }
    for( const auto& trk : output_trks2 )
    {
      if( trk->id() > max_output_id )
        max_output_id = trk->id();
    }

    kv::track_id_t next_unmatched_id = max_output_id + 1;

    for( const auto& pair : m_accumulated_tracks1 )
    {
      if( proc_left.find( pair.first ) == proc_left.end() )
      {
        auto cloned = pair.second->clone();
        cloned->set_id( next_unmatched_id++ );
        output_trks1.push_back( cloned );
      }
    }

    for( const auto& pair : m_accumulated_tracks2 )
    {
      if( proc_right.find( pair.first ) == proc_right.end() )
      {
        auto cloned = pair.second->clone();
        cloned->set_id( next_unmatched_id++ );
        output_trks2.push_back( cloned );
      }
    }
  }

  // Apply class averaging across matched stereo pairs if enabled
  if( m_average_stereo_classes )
  {
    std::map< kv::track_id_t, kv::track_sptr > left_by_id, right_by_id;
    for( const auto& trk : output_trks1 )
      left_by_id[trk->id()] = trk;
    for( const auto& trk : output_trks2 )
      right_by_id[trk->id()] = trk;

    bool weighted = use_weighted_averaging();
    bool sbc = use_scaled_by_conf();
    for( auto& entry : left_by_id )
    {
      auto rit = right_by_id.find( entry.first );
      if( rit == right_by_id.end() )
        continue;

      std::vector< kv::detected_object_sptr > left_dets, right_dets;
      for( auto it = entry.second->begin(); it != entry.second->end(); ++it )
      {
        auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
        if( ots && ots->detection() )
          left_dets.push_back( ots->detection() );
      }
      for( auto it = rit->second->begin(); it != rit->second->end(); ++it )
      {
        auto ots = std::dynamic_pointer_cast< kv::object_track_state >( *it );
        if( ots && ots->detection() )
          right_dets.push_back( ots->detection() );
      }

      auto avg_dot = compute_stereo_average_classification(
        left_dets, right_dets, weighted, sbc, m_class_averaging_ignore_class );
      if( avg_dot )
      {
        apply_classification_to_track( entry.second, avg_dot );
        apply_classification_to_track( rit->second, avg_dot );
      }
    }
  }

  // Apply track filtering
  output_trks1 = filter_tracks( output_trks1 );
  output_trks2 = filter_tracks( output_trks2 );
}

} // end namespace core
} // end namespace viame
