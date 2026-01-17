/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "evaluate_models.h"

#include <vital/logger/logger.h>

#include <filesystem>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace viame {

namespace kv = kwiver::vital;

// =============================================================================
// Helper structures for internal processing

/// Represents a single detection or ground truth annotation
struct detection
{
  int id = -1;                    // Detection/annotation ID
  int track_id = -1;              // Track ID (-1 if not a track)
  int frame_id = -1;              // Frame number
  std::string frame_name;         // Image/video identifier
  double x1 = 0, y1 = 0;          // Top-left corner
  double x2 = 0, y2 = 0;          // Bottom-right corner
  double confidence = 1.0;        // Detection confidence
  std::string class_name;         // Primary class label
  double class_confidence = 1.0;  // Confidence for primary class

  double width() const { return x2 - x1; }
  double height() const { return y2 - y1; }
  double area() const { return width() * height(); }
  double cx() const { return ( x1 + x2 ) / 2.0; }
  double cy() const { return ( y1 + y2 ) / 2.0; }
};

/// Match between a computed detection and ground truth
struct detection_match
{
  int computed_idx = -1;
  int gt_idx = -1;
  double iou = 0.0;
};

/// Per-frame matching results
struct frame_matches
{
  std::vector< detection_match > matches;
  std::vector< int > false_positives;   // Indices into computed
  std::vector< int > false_negatives;   // Indices into ground truth
};

/// Track association for ID metrics
struct track_association
{
  int gt_track_id = -1;
  int computed_track_id = -1;
  int num_matches = 0;
};

// =============================================================================
// Private implementation class

class model_evaluator::priv
{
public:
  priv()
    : m_logger( kv::get_logger( "viame.core.evaluate_models" ) )
  {}

  // Configuration
  evaluation_config m_config;

  // Logger
  kv::logger_handle_t m_logger;

  // Loaded data
  std::vector< detection > m_computed;
  std::vector< detection > m_groundtruth;

  // -------------------------------------------------------------------------
  // Cached data for performance optimization

  /// Frame indices for computed detections: frame_id -> [indices into m_computed]
  std::map< int, std::vector< size_t > > m_computed_by_frame;

  /// Frame indices for ground truth: frame_id -> [indices into m_groundtruth]
  std::map< int, std::vector< size_t > > m_gt_by_frame;

  /// Sorted list of all frame IDs
  std::vector< int > m_frame_list;

  /// Per-frame IoU matrices: frame_id -> [computed_idx][gt_idx] -> IoU
  std::map< int, std::vector< std::vector< double > > > m_iou_matrices;

  /// Per-frame matching results
  std::map< int, frame_matches > m_frame_matches;

  /// Track length caches
  std::map< int, int > m_gt_track_lengths;
  std::map< int, int > m_comp_track_lengths;

  /// Unique track IDs
  std::set< int > m_gt_track_ids;
  std::set< int > m_comp_track_ids;

  // -------------------------------------------------------------------------
  // File parsing

  bool parse_viame_csv( const std::string& filepath,
                        std::vector< detection >& detections );

  // -------------------------------------------------------------------------
  // Matching and caching

  double compute_iou( const detection& a, const detection& b ) const;

  void match_frame_with_matrix(
    const std::vector< std::vector< double > >& iou_matrix,
    size_t num_computed, size_t num_gt,
    double iou_threshold,
    frame_matches& matches );

  void match_frame(
    const std::vector< detection >& computed,
    const std::vector< detection >& groundtruth,
    frame_matches& matches );

  void build_caches();
  void perform_matching();
  void clear_caches();

  // -------------------------------------------------------------------------
  // Metric computation

  void compute_detection_metrics( evaluation_results& results );
  void compute_localization_metrics( evaluation_results& results );
  void compute_mot_metrics( evaluation_results& results );
  void compute_hota_metrics( evaluation_results& results );
  void compute_kwant_metrics( evaluation_results& results );
  void compute_track_quality_metrics( evaluation_results& results );
  void compute_average_precision( evaluation_results& results );
  void compute_multi_threshold_ap( evaluation_results& results );
  void compute_classification_metrics( evaluation_results& results );
  void compute_per_class_metrics( evaluation_results& results );
};

// =============================================================================
// CSV parsing

bool
model_evaluator::priv::parse_viame_csv( const std::string& filepath,
                                         std::vector< detection >& detections )
{
  std::ifstream file( filepath );
  if( !file.is_open() )
  {
    LOG_ERROR( m_logger, "Failed to open file: " << filepath );
    return false;
  }

  std::string line;
  int line_num = 0;

  while( std::getline( file, line ) )
  {
    line_num++;

    // Skip empty lines and comments
    if( line.empty() || line[0] == '#' )
    {
      continue;
    }

    // Parse CSV line
    std::vector< std::string > tokens;
    std::stringstream ss( line );
    std::string token;

    while( std::getline( ss, token, ',' ) )
    {
      // Trim whitespace
      size_t start = token.find_first_not_of( " \t" );
      size_t end = token.find_last_not_of( " \t" );
      if( start != std::string::npos )
      {
        token = token.substr( start, end - start + 1 );
      }
      tokens.push_back( token );
    }

    // Minimum 9 columns required for VIAME CSV
    if( tokens.size() < 9 )
    {
      LOG_WARN( m_logger, "Skipping malformed line " << line_num
                << " in " << filepath << " (only " << tokens.size() << " columns)" );
      continue;
    }

    try
    {
      detection det;

      // Column 0: Detection or Track ID
      det.id = std::stoi( tokens[0] );
      det.track_id = det.id;  // In VIAME CSV, ID is typically track ID

      // Column 1: Video or Image Identifier
      det.frame_name = tokens[1];

      // Column 2: Frame number
      det.frame_id = std::stoi( tokens[2] );

      // Columns 3-6: Bounding box (x1, y1, x2, y2)
      det.x1 = std::stod( tokens[3] );
      det.y1 = std::stod( tokens[4] );
      det.x2 = std::stod( tokens[5] );
      det.y2 = std::stod( tokens[6] );

      // Column 7: Detection confidence
      det.confidence = std::stod( tokens[7] );

      // Column 8: Target length (skip for now)

      // Columns 9+: Class/confidence pairs
      if( tokens.size() >= 11 )
      {
        det.class_name = tokens[9];
        det.class_confidence = std::stod( tokens[10] );
      }
      else if( tokens.size() >= 10 )
      {
        det.class_name = tokens[9];
        det.class_confidence = det.confidence;
      }
      else
      {
        det.class_name = "unknown";
        det.class_confidence = det.confidence;
      }

      // Apply confidence threshold
      if( det.confidence >= m_config.confidence_threshold )
      {
        detections.push_back( det );
      }
    }
    catch( const std::exception& e )
    {
      LOG_WARN( m_logger, "Failed to parse line " << line_num
                << " in " << filepath << ": " << e.what() );
    }
  }

  LOG_INFO( m_logger, "Loaded " << detections.size()
            << " detections from " << filepath );

  return true;
}

// =============================================================================
// IoU computation

double
model_evaluator::priv::compute_iou( const detection& a, const detection& b ) const
{
  // Compute intersection
  double ix1 = std::max( a.x1, b.x1 );
  double iy1 = std::max( a.y1, b.y1 );
  double ix2 = std::min( a.x2, b.x2 );
  double iy2 = std::min( a.y2, b.y2 );

  double iw = std::max( 0.0, ix2 - ix1 );
  double ih = std::max( 0.0, iy2 - iy1 );
  double intersection = iw * ih;

  // Compute union
  double area_a = a.area();
  double area_b = b.area();
  double union_area = area_a + area_b - intersection;

  if( union_area <= 0.0 )
  {
    return 0.0;
  }

  return intersection / union_area;
}

// =============================================================================
// Frame-level matching using pre-computed IoU matrix

void
model_evaluator::priv::match_frame_with_matrix(
  const std::vector< std::vector< double > >& iou_matrix,
  size_t num_computed, size_t num_gt,
  double iou_threshold,
  frame_matches& matches )
{
  matches.matches.clear();
  matches.false_positives.clear();
  matches.false_negatives.clear();

  if( num_computed == 0 && num_gt == 0 )
  {
    return;
  }

  if( num_computed == 0 )
  {
    matches.false_negatives.reserve( num_gt );
    for( size_t i = 0; i < num_gt; i++ )
    {
      matches.false_negatives.push_back( static_cast< int >( i ) );
    }
    return;
  }

  if( num_gt == 0 )
  {
    matches.false_positives.reserve( num_computed );
    for( size_t i = 0; i < num_computed; i++ )
    {
      matches.false_positives.push_back( static_cast< int >( i ) );
    }
    return;
  }

  // Greedy matching using pre-computed IoU matrix
  std::vector< bool > computed_matched( num_computed, false );
  std::vector< bool > gt_matched( num_gt, false );

  // Create sorted list of all potential matches
  std::vector< std::tuple< double, int, int > > potential_matches;
  potential_matches.reserve( num_computed * num_gt / 4 );  // Estimate

  for( size_t i = 0; i < num_computed; i++ )
  {
    for( size_t j = 0; j < num_gt; j++ )
    {
      if( iou_matrix[i][j] >= iou_threshold )
      {
        potential_matches.emplace_back( iou_matrix[i][j], i, j );
      }
    }
  }

  // Sort by IoU descending
  std::sort( potential_matches.begin(), potential_matches.end(),
    []( const auto& a, const auto& b ) { return std::get<0>( a ) > std::get<0>( b ); } );

  // Greedily assign matches
  matches.matches.reserve( std::min( num_computed, num_gt ) );

  for( const auto& pm : potential_matches )
  {
    int ci = std::get<1>( pm );
    int gi = std::get<2>( pm );

    if( !computed_matched[ci] && !gt_matched[gi] )
    {
      detection_match dm;
      dm.computed_idx = ci;
      dm.gt_idx = gi;
      dm.iou = std::get<0>( pm );
      matches.matches.push_back( dm );

      computed_matched[ci] = true;
      gt_matched[gi] = true;
    }
  }

  // Collect false positives (unmatched computed)
  for( size_t i = 0; i < num_computed; i++ )
  {
    if( !computed_matched[i] )
    {
      matches.false_positives.push_back( static_cast< int >( i ) );
    }
  }

  // Collect false negatives (unmatched ground truth)
  for( size_t j = 0; j < num_gt; j++ )
  {
    if( !gt_matched[j] )
    {
      matches.false_negatives.push_back( static_cast< int >( j ) );
    }
  }
}

// =============================================================================
// Match detections for a single frame (computes IoU on the fly)
// Used for per-class metrics where we don't have pre-computed IoU matrices

void
model_evaluator::priv::match_frame(
  const std::vector< detection >& computed,
  const std::vector< detection >& groundtruth,
  frame_matches& matches )
{
  size_t num_comp = computed.size();
  size_t num_gt = groundtruth.size();

  // Compute IoU matrix for this frame
  std::vector< std::vector< double > > iou_matrix;
  if( num_comp > 0 && num_gt > 0 )
  {
    iou_matrix.resize( num_comp, std::vector< double >( num_gt, 0.0 ) );
    for( size_t ci = 0; ci < num_comp; ci++ )
    {
      for( size_t gi = 0; gi < num_gt; gi++ )
      {
        iou_matrix[ci][gi] = compute_iou( computed[ci], groundtruth[gi] );
      }
    }
  }

  // Delegate to matrix-based matching
  match_frame_with_matrix( iou_matrix, num_comp, num_gt,
                           m_config.iou_threshold, matches );
}

// =============================================================================
// Build cached data structures for efficient processing

void
model_evaluator::priv::build_caches()
{
  // Clear existing caches
  clear_caches();

  // Build frame groupings
  for( size_t i = 0; i < m_computed.size(); i++ )
  {
    int frame_id = m_computed[i].frame_id;
    m_computed_by_frame[frame_id].push_back( i );

    // Cache track info
    int track_id = m_computed[i].track_id;
    if( track_id >= 0 )
    {
      m_comp_track_lengths[track_id]++;
      m_comp_track_ids.insert( track_id );
    }
  }

  for( size_t i = 0; i < m_groundtruth.size(); i++ )
  {
    int frame_id = m_groundtruth[i].frame_id;
    m_gt_by_frame[frame_id].push_back( i );

    // Cache track info
    int track_id = m_groundtruth[i].track_id;
    if( track_id >= 0 )
    {
      m_gt_track_lengths[track_id]++;
      m_gt_track_ids.insert( track_id );
    }
  }

  // Build sorted frame list
  std::set< int > all_frames;
  for( const auto& p : m_computed_by_frame )
  {
    all_frames.insert( p.first );
  }
  for( const auto& p : m_gt_by_frame )
  {
    all_frames.insert( p.first );
  }
  m_frame_list.assign( all_frames.begin(), all_frames.end() );

  // Pre-compute IoU matrices for all frames
  m_iou_matrices.clear();

  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic)
  #endif
  for( int fi = 0; fi < static_cast<int>( m_frame_list.size() ); fi++ )
  {
    int frame_id = m_frame_list[fi];

    auto comp_it = m_computed_by_frame.find( frame_id );
    auto gt_it = m_gt_by_frame.find( frame_id );

    size_t num_comp = ( comp_it != m_computed_by_frame.end() ) ?
                      comp_it->second.size() : 0;
    size_t num_gt = ( gt_it != m_gt_by_frame.end() ) ?
                    gt_it->second.size() : 0;

    std::vector< std::vector< double > > iou_matrix(
      num_comp, std::vector< double >( num_gt, 0.0 ) );

    if( num_comp > 0 && num_gt > 0 )
    {
      const auto& comp_indices = comp_it->second;
      const auto& gt_indices = gt_it->second;

      for( size_t i = 0; i < num_comp; i++ )
      {
        for( size_t j = 0; j < num_gt; j++ )
        {
          iou_matrix[i][j] = compute_iou(
            m_computed[comp_indices[i]],
            m_groundtruth[gt_indices[j]] );
        }
      }
    }

    #ifdef _OPENMP
    #pragma omp critical
    #endif
    {
      m_iou_matrices[frame_id] = std::move( iou_matrix );
    }
  }

  LOG_INFO( m_logger, "Built caches: " << m_frame_list.size() << " frames, "
            << m_comp_track_ids.size() << " computed tracks, "
            << m_gt_track_ids.size() << " GT tracks" );
}

void
model_evaluator::priv::clear_caches()
{
  m_computed_by_frame.clear();
  m_gt_by_frame.clear();
  m_frame_list.clear();
  m_iou_matrices.clear();
  m_frame_matches.clear();
  m_gt_track_lengths.clear();
  m_comp_track_lengths.clear();
  m_gt_track_ids.clear();
  m_comp_track_ids.clear();
}

// =============================================================================
// Perform matching using cached IoU matrices

void
model_evaluator::priv::perform_matching()
{
  m_frame_matches.clear();

  #ifdef _OPENMP
  std::vector< std::pair< int, frame_matches > > results( m_frame_list.size() );

  #pragma omp parallel for schedule(dynamic)
  for( size_t fi = 0; fi < m_frame_list.size(); fi++ )
  {
    int frame_id = m_frame_list[fi];

    auto comp_it = m_computed_by_frame.find( frame_id );
    auto gt_it = m_gt_by_frame.find( frame_id );

    size_t num_comp = ( comp_it != m_computed_by_frame.end() ) ?
                      comp_it->second.size() : 0;
    size_t num_gt = ( gt_it != m_gt_by_frame.end() ) ?
                    gt_it->second.size() : 0;

    frame_matches fm;
    auto iou_it = m_iou_matrices.find( frame_id );
    if( iou_it != m_iou_matrices.end() )
    {
      match_frame_with_matrix( iou_it->second, num_comp, num_gt,
                               m_config.iou_threshold, fm );
    }
    else
    {
      match_frame_with_matrix( {}, num_comp, num_gt,
                               m_config.iou_threshold, fm );
    }

    results[fi] = { frame_id, std::move( fm ) };
  }

  // Collect results
  for( auto& r : results )
  {
    m_frame_matches[r.first] = std::move( r.second );
  }

  #else
  // Non-OpenMP version
  for( int frame_id : m_frame_list )
  {
    auto comp_it = m_computed_by_frame.find( frame_id );
    auto gt_it = m_gt_by_frame.find( frame_id );

    size_t num_comp = ( comp_it != m_computed_by_frame.end() ) ?
                      comp_it->second.size() : 0;
    size_t num_gt = ( gt_it != m_gt_by_frame.end() ) ?
                    gt_it->second.size() : 0;

    frame_matches fm;
    auto iou_it = m_iou_matrices.find( frame_id );
    if( iou_it != m_iou_matrices.end() )
    {
      match_frame_with_matrix( iou_it->second, num_comp, num_gt,
                               m_config.iou_threshold, fm );
    }

    m_frame_matches[frame_id] = std::move( fm );
  }
  #endif
}

// =============================================================================
// Detection metrics computation

void
model_evaluator::priv::compute_detection_metrics( evaluation_results& results )
{
  double tp = 0, fp = 0, fn = 0;
  double total_iou = 0;

  for( const auto& fm_pair : m_frame_matches )
  {
    const auto& fm = fm_pair.second;
    tp += fm.matches.size();
    fp += fm.false_positives.size();
    fn += fm.false_negatives.size();

    for( const auto& m : fm.matches )
    {
      total_iou += m.iou;
    }
  }

  results.true_positives = tp;
  results.false_positives = fp;
  results.false_negatives = fn;

  // Precision
  if( tp + fp > 0 )
  {
    results.precision = tp / ( tp + fp );
  }

  // Recall
  if( tp + fn > 0 )
  {
    results.recall = tp / ( tp + fn );
  }

  // F1 Score
  if( results.precision + results.recall > 0 )
  {
    results.f1_score = 2.0 * results.precision * results.recall /
                       ( results.precision + results.recall );
  }

  // False Discovery Rate = FP / (FP + TP)
  if( fp + tp > 0 )
  {
    results.false_discovery_rate = fp / ( fp + tp );
  }

  // Miss Rate = FN / (FN + TP)
  if( fn + tp > 0 )
  {
    results.miss_rate = fn / ( fn + tp );
  }

  // Matthews Correlation Coefficient
  // MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
  // For detection, TN is typically undefined or very large
  // We use a simplified version: MCC = sqrt(precision * recall) - sqrt((1-precision)*(1-recall))
  // Or the more standard: using total negatives estimation
  double tn = 0;  // True negatives not typically computed for detection
  double mcc_denom = std::sqrt( ( tp + fp ) * ( tp + fn ) * ( tn + fp ) * ( tn + fn ) );
  if( mcc_denom > 0 )
  {
    results.mcc = ( tp * tn - fp * fn ) / mcc_denom;
  }
  else
  {
    // Alternative MCC computation without TN
    // Based on precision and recall
    if( results.precision > 0 && results.recall > 0 )
    {
      // Phi coefficient approximation
      double ppv = results.precision;
      double tpr = results.recall;
      // FDR = 1 - precision, FNR = 1 - recall
      results.mcc = std::sqrt( ppv * tpr ) -
                    std::sqrt( ( 1.0 - ppv ) * ( 1.0 - tpr ) );
    }
  }

  // MOTP (average IoU on matches)
  if( tp > 0 )
  {
    results.motp = total_iou / tp;
  }

  // Statistics (use cached data)
  results.total_gt_objects = m_groundtruth.size();
  results.total_computed = m_computed.size();
  results.total_frames = m_frame_list.size();
  results.total_gt_tracks = m_gt_track_ids.size();
  results.total_computed_tracks = m_comp_track_ids.size();
}

// =============================================================================
// Localization quality metrics

void
model_evaluator::priv::compute_localization_metrics( evaluation_results& results )
{
  std::vector< double > ious;
  std::vector< double > center_distances;
  std::vector< double > size_errors;

  // Reserve space based on expected number of matches
  size_t expected_matches = std::min( m_computed.size(), m_groundtruth.size() );
  ious.reserve( expected_matches );
  center_distances.reserve( expected_matches );
  size_errors.reserve( expected_matches );

  for( const auto& fm_pair : m_frame_matches )
  {
    int frame_id = fm_pair.first;
    const auto& fm = fm_pair.second;

    auto comp_it = m_computed_by_frame.find( frame_id );
    auto gt_it = m_gt_by_frame.find( frame_id );

    if( comp_it == m_computed_by_frame.end() || gt_it == m_gt_by_frame.end() )
    {
      continue;
    }

    const auto& comp_indices = comp_it->second;
    const auto& gt_indices = gt_it->second;

    for( const auto& m : fm.matches )
    {
      if( m.computed_idx < 0 || m.gt_idx < 0 )
      {
        continue;
      }
      if( static_cast< size_t >( m.computed_idx ) >= comp_indices.size() ||
          static_cast< size_t >( m.gt_idx ) >= gt_indices.size() )
      {
        continue;
      }

      // Direct access to original data using cached indices
      const auto& comp = m_computed[comp_indices[m.computed_idx]];
      const auto& gt = m_groundtruth[gt_indices[m.gt_idx]];

      // IoU
      ious.push_back( m.iou );

      // Center distance
      double dx = comp.cx() - gt.cx();
      double dy = comp.cy() - gt.cy();
      center_distances.push_back( std::sqrt( dx * dx + dy * dy ) );

      // Relative size error: |area_comp - area_gt| / area_gt
      double gt_area = gt.area();
      if( gt_area > 0 )
      {
        double size_err = std::abs( comp.area() - gt_area ) / gt_area;
        size_errors.push_back( size_err );
      }
    }
  }

  // Mean IoU
  if( !ious.empty() )
  {
    double sum = std::accumulate( ious.begin(), ious.end(), 0.0 );
    results.mean_iou = sum / ious.size();

    // Median IoU
    std::vector< double > sorted_ious = ious;
    std::sort( sorted_ious.begin(), sorted_ious.end() );
    size_t n = sorted_ious.size();
    if( n % 2 == 0 )
    {
      results.median_iou = ( sorted_ious[n / 2 - 1] + sorted_ious[n / 2] ) / 2.0;
    }
    else
    {
      results.median_iou = sorted_ious[n / 2];
    }
  }

  // Mean center distance
  if( !center_distances.empty() )
  {
    double sum = std::accumulate( center_distances.begin(), center_distances.end(), 0.0 );
    results.mean_center_distance = sum / center_distances.size();
  }

  // Mean size error
  if( !size_errors.empty() )
  {
    double sum = std::accumulate( size_errors.begin(), size_errors.end(), 0.0 );
    results.mean_size_error = sum / size_errors.size();
  }
}

// =============================================================================
// MOT metrics computation

void
model_evaluator::priv::compute_mot_metrics( evaluation_results& results )
{
  if( !m_config.compute_tracking_metrics )
  {
    return;
  }

  double tp = results.true_positives;
  double fp = results.false_positives;
  double fn = results.false_negatives;

  // Get sorted frame list
  std::vector< int > frame_list;
  for( const auto& fm_pair : m_frame_matches )
  {
    frame_list.push_back( fm_pair.first );
  }
  std::sort( frame_list.begin(), frame_list.end() );

  // Track assignment history: gt_track_id -> last computed_track_id
  std::map< int, int > gt_to_computed_assignment;
  // Reverse mapping: computed_track_id -> last gt_track_id
  std::map< int, int > computed_to_gt_assignment;

  // ID switches and fragmentations
  double id_switches = 0;
  double fragmentations = 0;

  // pymotmetrics extended ID switch decomposition
  double num_transfer = 0;  // Computed track transfers to different GT
  double num_ascend = 0;    // Computed track ascends to take over tracking of a GT
  double num_migrate = 0;   // GT migrates to a different computed track

  // Build frame-indexed lookup for original detections
  // Use cached frame groupings
  const auto& computed_by_frame = m_computed_by_frame;
  const auto& gt_by_frame = m_gt_by_frame;

  // Process frames in order
  std::map< int, bool > gt_track_was_matched;  // For fragmentation detection

  for( int frame_id : frame_list )
  {
    const auto& fm = m_frame_matches[frame_id];

    auto comp_it = computed_by_frame.find( frame_id );
    auto gt_it = gt_by_frame.find( frame_id );

    std::vector< detection > frame_computed;
    std::vector< detection > frame_gt;

    if( comp_it != computed_by_frame.end() )
    {
      for( size_t idx : comp_it->second )
      {
        frame_computed.push_back( m_computed[idx] );
      }
    }
    if( gt_it != gt_by_frame.end() )
    {
      for( size_t idx : gt_it->second )
      {
        frame_gt.push_back( m_groundtruth[idx] );
      }
    }

    for( const auto& m : fm.matches )
    {
      if( m.computed_idx < 0 || m.gt_idx < 0 )
      {
        continue;
      }
      if( static_cast< size_t >( m.computed_idx ) >= frame_computed.size() ||
          static_cast< size_t >( m.gt_idx ) >= frame_gt.size() )
      {
        continue;
      }

      int gt_track = frame_gt[m.gt_idx].track_id;
      int comp_track = frame_computed[m.computed_idx].track_id;

      if( gt_track < 0 || comp_track < 0 )
      {
        continue;
      }

      // Get previous assignments if they exist
      auto prev_comp_it = gt_to_computed_assignment.find( gt_track );
      auto prev_gt_it = computed_to_gt_assignment.find( comp_track );

      int prev_comp = ( prev_comp_it != gt_to_computed_assignment.end() ) ?
                      prev_comp_it->second : -1;
      int prev_gt = ( prev_gt_it != computed_to_gt_assignment.end() ) ?
                    prev_gt_it->second : -1;

      // Check for ID switch (from GT perspective - migrate)
      if( prev_comp >= 0 && prev_comp != comp_track )
      {
        id_switches++;
        num_migrate++;  // GT migrated to a different computed track

        // Ascend: computed track takes over from another on the same GT
        // This happens when the computed track already existed (was tracking something else)
        if( prev_gt >= 0 )
        {
          num_ascend++;
        }
      }

      // Transfer: computed track switches from one GT to another
      if( prev_gt >= 0 && prev_gt != gt_track )
      {
        num_transfer++;
      }

      // Check for fragmentation (was matched, then not matched, now matched again)
      auto was_matched_it = gt_track_was_matched.find( gt_track );
      if( was_matched_it != gt_track_was_matched.end() && !was_matched_it->second )
      {
        fragmentations++;
      }

      // Update both assignment maps
      gt_to_computed_assignment[gt_track] = comp_track;
      computed_to_gt_assignment[comp_track] = gt_track;
      gt_track_was_matched[gt_track] = true;
    }

    // Mark unmatched GT tracks as not matched this frame
    for( int fn_idx : fm.false_negatives )
    {
      if( static_cast< size_t >( fn_idx ) < frame_gt.size() )
      {
        int gt_track = frame_gt[fn_idx].track_id;
        if( gt_track >= 0 && gt_track_was_matched.count( gt_track ) )
        {
          gt_track_was_matched[gt_track] = false;
        }
      }
    }
  }

  results.id_switches = id_switches;
  results.fragmentations = fragmentations;
  results.num_transfer = num_transfer;
  results.num_ascend = num_ascend;
  results.num_migrate = num_migrate;

  // MOTA = 1 - (FN + FP + ID_switches) / total_gt_objects
  double total_gt = tp + fn;  // Total ground truth detections
  if( total_gt > 0 )
  {
    results.mota = 1.0 - ( fn + fp + id_switches ) / total_gt;
  }

  // -------------------------------------------------------------------------
  // ID metrics (IDF1, IDP, IDR) - Global ID assignment

  // Build global association: count matches per (gt_track, computed_track) pair
  std::map< std::pair< int, int >, int > track_pair_counts;
  std::map< int, int > gt_track_total;
  std::map< int, int > comp_track_total;

  for( int frame_id : frame_list )
  {
    const auto& fm = m_frame_matches[frame_id];

    auto comp_it = computed_by_frame.find( frame_id );
    auto gt_it = gt_by_frame.find( frame_id );

    std::vector< detection > frame_computed;
    std::vector< detection > frame_gt;

    if( comp_it != computed_by_frame.end() )
    {
      for( size_t idx : comp_it->second )
      {
        frame_computed.push_back( m_computed[idx] );
      }
    }
    if( gt_it != gt_by_frame.end() )
    {
      for( size_t idx : gt_it->second )
      {
        frame_gt.push_back( m_groundtruth[idx] );
      }
    }

    // Count GT track occurrences
    for( const auto& d : frame_gt )
    {
      if( d.track_id >= 0 )
      {
        gt_track_total[d.track_id]++;
      }
    }

    // Count computed track occurrences
    for( const auto& d : frame_computed )
    {
      if( d.track_id >= 0 )
      {
        comp_track_total[d.track_id]++;
      }
    }

    // Count matches per track pair
    for( const auto& m : fm.matches )
    {
      if( m.computed_idx < 0 || m.gt_idx < 0 )
      {
        continue;
      }
      if( static_cast< size_t >( m.computed_idx ) >= frame_computed.size() ||
          static_cast< size_t >( m.gt_idx ) >= frame_gt.size() )
      {
        continue;
      }

      int gt_track = frame_gt[m.gt_idx].track_id;
      int comp_track = frame_computed[m.computed_idx].track_id;

      if( gt_track >= 0 && comp_track >= 0 )
      {
        track_pair_counts[{ gt_track, comp_track }]++;
      }
    }
  }

  // Find optimal global assignment (greedy approximation)
  std::vector< std::tuple< int, int, int > > sorted_pairs;
  for( const auto& p : track_pair_counts )
  {
    sorted_pairs.emplace_back( p.second, p.first.first, p.first.second );
  }
  std::sort( sorted_pairs.begin(), sorted_pairs.end(),
    []( const auto& a, const auto& b ) { return std::get<0>( a ) > std::get<0>( b ); } );

  std::set< int > assigned_gt, assigned_comp;
  double idtp = 0;

  for( const auto& sp : sorted_pairs )
  {
    int count = std::get<0>( sp );
    int gt_id = std::get<1>( sp );
    int comp_id = std::get<2>( sp );

    if( assigned_gt.count( gt_id ) == 0 && assigned_comp.count( comp_id ) == 0 )
    {
      idtp += count;
      assigned_gt.insert( gt_id );
      assigned_comp.insert( comp_id );
    }
  }

  double total_gt_detections = 0;
  for( const auto& p : gt_track_total )
  {
    total_gt_detections += p.second;
  }

  double total_comp_detections = 0;
  for( const auto& p : comp_track_total )
  {
    total_comp_detections += p.second;
  }

  double idfn = total_gt_detections - idtp;
  double idfp = total_comp_detections - idtp;

  // IDP = IDTP / (IDTP + IDFP)
  if( idtp + idfp > 0 )
  {
    results.idp = idtp / ( idtp + idfp );
  }

  // IDR = IDTP / (IDTP + IDFN)
  if( idtp + idfn > 0 )
  {
    results.idr = idtp / ( idtp + idfn );
  }

  // IDF1 = 2 * IDP * IDR / (IDP + IDR)
  if( results.idp + results.idr > 0 )
  {
    results.idf1 = 2.0 * results.idp * results.idr / ( results.idp + results.idr );
  }

  // -------------------------------------------------------------------------
  // Track quality metrics (mostly tracked/partially tracked/mostly lost)

  std::map< int, int > gt_track_matched_frames;
  std::map< int, int > gt_track_total_frames;

  for( int frame_id : frame_list )
  {
    auto gt_it = gt_by_frame.find( frame_id );
    if( gt_it == gt_by_frame.end() )
    {
      continue;
    }

    std::vector< detection > frame_gt;
    for( size_t idx : gt_it->second )
    {
      frame_gt.push_back( m_groundtruth[idx] );
    }

    // Count total frames for each GT track
    for( const auto& d : frame_gt )
    {
      if( d.track_id >= 0 )
      {
        gt_track_total_frames[d.track_id]++;
      }
    }

    // Count matched frames for each GT track
    const auto& fm = m_frame_matches[frame_id];
    for( const auto& m : fm.matches )
    {
      if( m.gt_idx >= 0 && static_cast< size_t >( m.gt_idx ) < frame_gt.size() )
      {
        int gt_track = frame_gt[m.gt_idx].track_id;
        if( gt_track >= 0 )
        {
          gt_track_matched_frames[gt_track]++;
        }
      }
    }
  }

  double mostly_tracked = 0, partially_tracked = 0, mostly_lost = 0;

  for( const auto& p : gt_track_total_frames )
  {
    int gt_track = p.first;
    int total = p.second;
    int matched = gt_track_matched_frames[gt_track];

    double ratio = static_cast< double >( matched ) / total;

    if( ratio >= 0.8 )
    {
      mostly_tracked++;
    }
    else if( ratio >= 0.2 )
    {
      partially_tracked++;
    }
    else
    {
      mostly_lost++;
    }
  }

  results.mostly_tracked = mostly_tracked;
  results.partially_tracked = partially_tracked;
  results.mostly_lost = mostly_lost;

  // Normalized rates (as fractions)
  double total_tracks = mostly_tracked + partially_tracked + mostly_lost;
  if( total_tracks > 0 )
  {
    results.mt_ratio = mostly_tracked / total_tracks;
    results.pt_ratio = partially_tracked / total_tracks;
    results.ml_ratio = mostly_lost / total_tracks;
  }

  // False Alarms per Frame
  if( results.total_frames > 0 )
  {
    results.faf = results.false_positives / results.total_frames;
  }
}

// =============================================================================
// HOTA metrics computation (Higher Order Tracking Accuracy)

void
model_evaluator::priv::compute_hota_metrics( evaluation_results& results )
{
  if( !m_config.compute_tracking_metrics )
  {
    return;
  }

  // HOTA is computed over multiple IoU thresholds (0.05 to 0.95, step 0.05)
  // For each threshold:
  //   DetA = |TP| / (|TP| + |FN| + |FP|)
  //   AssA = (1/|TP|) * sum over TPs of |TPA| / (|TPA| + |FNA| + |FPA|)
  //   HOTA = sqrt(DetA * AssA)
  // Final HOTA = average over all thresholds

  std::vector< double > iou_thresholds;
  for( double t = 0.05; t <= 0.95; t += 0.05 )
  {
    iou_thresholds.push_back( t );
  }

  // Use cached track appearances (already computed in build_caches)
  const auto& gt_track_appearances = m_gt_track_lengths;
  const auto& comp_track_appearances = m_comp_track_lengths;

  double sum_hota = 0.0;
  double sum_deta = 0.0;
  double sum_assa = 0.0;
  double sum_loca = 0.0;
  int num_thresholds = 0;

  // Process each threshold (can be parallelized)
  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:sum_hota,sum_deta,sum_assa,sum_loca,num_thresholds)
  #endif
  for( size_t ti = 0; ti < iou_thresholds.size(); ti++ )
  {
    double alpha = iou_thresholds[ti];

    // Re-match at this IoU threshold using cached IoU matrices
    double tp_alpha = 0, fp_alpha = 0, fn_alpha = 0;
    double total_iou_alpha = 0;

    // For AssA: track which GT and computed tracks are associated at each TP
    std::map< std::pair< int, int >, int > track_pair_tps;

    for( int frame_id : m_frame_list )
    {
      auto comp_it = m_computed_by_frame.find( frame_id );
      auto gt_it = m_gt_by_frame.find( frame_id );
      auto iou_it = m_iou_matrices.find( frame_id );

      size_t num_comp = ( comp_it != m_computed_by_frame.end() ) ?
                        comp_it->second.size() : 0;
      size_t num_gt = ( gt_it != m_gt_by_frame.end() ) ?
                      gt_it->second.size() : 0;

      if( num_comp == 0 && num_gt == 0 )
      {
        continue;
      }

      // Use cached IoU matrix
      const std::vector< std::vector< double > >* iou_matrix_ptr = nullptr;
      if( iou_it != m_iou_matrices.end() )
      {
        iou_matrix_ptr = &iou_it->second;
      }

      // Greedy matching at alpha threshold
      std::vector< bool > computed_matched( num_comp, false );
      std::vector< bool > gt_matched( num_gt, false );

      if( iou_matrix_ptr && num_comp > 0 && num_gt > 0 )
      {
        const auto& iou_matrix = *iou_matrix_ptr;

        std::vector< std::tuple< double, int, int > > potential_matches;
        potential_matches.reserve( num_comp );

        for( size_t i = 0; i < num_comp; i++ )
        {
          for( size_t j = 0; j < num_gt; j++ )
          {
            if( iou_matrix[i][j] >= alpha )
            {
              potential_matches.emplace_back( iou_matrix[i][j], i, j );
            }
          }
        }

        std::sort( potential_matches.begin(), potential_matches.end(),
          []( const auto& a, const auto& b ) { return std::get<0>( a ) > std::get<0>( b ); } );

        const auto& comp_indices = comp_it->second;
        const auto& gt_indices = gt_it->second;

        for( const auto& pm : potential_matches )
        {
          int ci = std::get<1>( pm );
          int gi = std::get<2>( pm );

          if( !computed_matched[ci] && !gt_matched[gi] )
          {
            tp_alpha++;
            total_iou_alpha += std::get<0>( pm );
            computed_matched[ci] = true;
            gt_matched[gi] = true;

            // Record track association using cached indices
            int gt_track = m_groundtruth[gt_indices[gi]].track_id;
            int comp_track = m_computed[comp_indices[ci]].track_id;
            if( gt_track >= 0 && comp_track >= 0 )
            {
              track_pair_tps[{ gt_track, comp_track }]++;
            }
          }
        }
      }

      // Count FPs and FNs
      for( size_t i = 0; i < num_comp; i++ )
      {
        if( !computed_matched[i] )
        {
          fp_alpha++;
        }
      }
      for( size_t j = 0; j < num_gt; j++ )
      {
        if( !gt_matched[j] )
        {
          fn_alpha++;
        }
      }
    }

    // Compute DetA
    double deta_alpha = 0.0;
    if( tp_alpha + fn_alpha + fp_alpha > 0 )
    {
      deta_alpha = tp_alpha / ( tp_alpha + fn_alpha + fp_alpha );
    }

    // Compute AssA
    // For each TP, compute its association score:
    // A(c) = |TPA(c)| / (|TPA(c)| + |FNA(c)| + |FPA(c)|)
    // where TPA(c) = TPs where c's GT matches, FNA = GT appearances not matched by c,
    // FPA = c's appearances not matched to its GT

    // First, find optimal GT assignment for each computed track (most matches)
    std::map< int, int > comp_to_gt_assignment;
    std::map< int, int > comp_to_gt_tps;

    for( const auto& p : track_pair_tps )
    {
      int gt_track = p.first.first;
      int comp_track = p.first.second;
      int count = p.second;

      auto it = comp_to_gt_tps.find( comp_track );
      if( it == comp_to_gt_tps.end() || count > it->second )
      {
        comp_to_gt_assignment[comp_track] = gt_track;
        comp_to_gt_tps[comp_track] = count;
      }
    }

    double sum_assa_alpha = 0.0;
    int assa_count = 0;

    // For each TP, compute association score
    for( const auto& p : track_pair_tps )
    {
      int gt_track = p.first.first;
      int comp_track = p.first.second;
      int tpa = p.second;  // TPs where this pair matched

      // Check if this is the assigned GT for this computed track
      auto assign_it = comp_to_gt_assignment.find( comp_track );
      if( assign_it == comp_to_gt_assignment.end() || assign_it->second != gt_track )
      {
        continue;  // Not the primary association
      }

      // FNA = GT appearances - TPA (GT frames not matched by this comp track)
      int gt_total = gt_track_appearances.at(gt_track);
      int fna = gt_total - tpa;

      // FPA = Comp appearances - TPA (Comp frames not matched to this GT)
      int comp_total = comp_track_appearances.at(comp_track);
      int fpa = comp_total - tpa;

      double a_score = 0.0;
      if( tpa + fna + fpa > 0 )
      {
        a_score = static_cast< double >( tpa ) / ( tpa + fna + fpa );
      }

      // Weight by number of TPs in this association
      sum_assa_alpha += a_score * tpa;
      assa_count += tpa;
    }

    double assa_alpha = 0.0;
    if( assa_count > 0 )
    {
      assa_alpha = sum_assa_alpha / assa_count;
    }

    // Compute HOTA
    double hota_alpha = std::sqrt( deta_alpha * assa_alpha );

    // Compute LocA (average IoU of TPs)
    double loca_alpha = 0.0;
    if( tp_alpha > 0 )
    {
      loca_alpha = total_iou_alpha / tp_alpha;
    }

    sum_hota += hota_alpha;
    sum_deta += deta_alpha;
    sum_assa += assa_alpha;
    sum_loca += loca_alpha;
    num_thresholds++;
  }

  // Average over all thresholds
  if( num_thresholds > 0 )
  {
    results.hota = sum_hota / num_thresholds;
    results.deta = sum_deta / num_thresholds;
    results.assa = sum_assa / num_thresholds;
    results.loca = sum_loca / num_thresholds;
  }
}

// =============================================================================
// KWANT-style metrics computation

void
model_evaluator::priv::compute_kwant_metrics( evaluation_results& results )
{
  if( !m_config.compute_tracking_metrics )
  {
    return;
  }

  // Get sorted frame list
  std::vector< int > frame_list;
  for( const auto& fm_pair : m_frame_matches )
  {
    frame_list.push_back( fm_pair.first );
  }
  std::sort( frame_list.begin(), frame_list.end() );

  // Use cached frame groupings
  const auto& computed_by_frame = m_computed_by_frame;
  const auto& gt_by_frame = m_gt_by_frame;

  // -------------------------------------------------------------------------
  // Track continuity: 1 / number_of_segments
  // A segment break occurs when a track is not matched for one or more frames

  std::map< int, std::vector< bool > > gt_track_matched_per_frame;
  std::map< int, std::vector< bool > > comp_track_matched_per_frame;

  // Initialize with all frames
  std::set< int > all_gt_tracks, all_comp_tracks;
  for( const auto& d : m_groundtruth )
  {
    if( d.track_id >= 0 )
    {
      all_gt_tracks.insert( d.track_id );
    }
  }
  for( const auto& d : m_computed )
  {
    if( d.track_id >= 0 )
    {
      all_comp_tracks.insert( d.track_id );
    }
  }

  // Build per-frame match status for each track
  for( int frame_id : frame_list )
  {
    const auto& fm = m_frame_matches[frame_id];

    auto comp_it = computed_by_frame.find( frame_id );
    auto gt_it = gt_by_frame.find( frame_id );

    std::vector< detection > frame_computed;
    std::vector< detection > frame_gt;

    if( comp_it != computed_by_frame.end() )
    {
      for( size_t idx : comp_it->second )
      {
        frame_computed.push_back( m_computed[idx] );
      }
    }
    if( gt_it != gt_by_frame.end() )
    {
      for( size_t idx : gt_it->second )
      {
        frame_gt.push_back( m_groundtruth[idx] );
      }
    }

    // Track which tracks are matched this frame
    std::set< int > matched_gt_tracks, matched_comp_tracks;
    for( const auto& m : fm.matches )
    {
      if( m.gt_idx >= 0 && static_cast< size_t >( m.gt_idx ) < frame_gt.size() )
      {
        matched_gt_tracks.insert( frame_gt[m.gt_idx].track_id );
      }
      if( m.computed_idx >= 0 && static_cast< size_t >( m.computed_idx ) < frame_computed.size() )
      {
        matched_comp_tracks.insert( frame_computed[m.computed_idx].track_id );
      }
    }

    // Record match status for GT tracks present in this frame
    for( const auto& d : frame_gt )
    {
      if( d.track_id >= 0 )
      {
        gt_track_matched_per_frame[d.track_id].push_back(
          matched_gt_tracks.count( d.track_id ) > 0 );
      }
    }

    // Record match status for computed tracks present in this frame
    for( const auto& d : frame_computed )
    {
      if( d.track_id >= 0 )
      {
        comp_track_matched_per_frame[d.track_id].push_back(
          matched_comp_tracks.count( d.track_id ) > 0 );
      }
    }
  }

  // Compute continuity for each track
  auto count_segments = []( const std::vector< bool >& matched_status ) -> int
  {
    if( matched_status.empty() )
    {
      return 0;
    }
    int segments = 0;
    bool in_segment = false;
    for( bool matched : matched_status )
    {
      if( matched && !in_segment )
      {
        segments++;
        in_segment = true;
      }
      else if( !matched )
      {
        in_segment = false;
      }
    }
    return std::max( segments, 1 );
  };

  double total_target_continuity = 0;
  int num_gt_tracks = 0;
  for( const auto& p : gt_track_matched_per_frame )
  {
    int segments = count_segments( p.second );
    if( segments > 0 )
    {
      total_target_continuity += 1.0 / segments;
      num_gt_tracks++;
    }
  }

  double total_track_continuity = 0;
  int num_comp_tracks = 0;
  for( const auto& p : comp_track_matched_per_frame )
  {
    int segments = count_segments( p.second );
    if( segments > 0 )
    {
      total_track_continuity += 1.0 / segments;
      num_comp_tracks++;
    }
  }

  if( num_gt_tracks > 0 )
  {
    results.avg_target_continuity = total_target_continuity / num_gt_tracks;
  }
  if( num_comp_tracks > 0 )
  {
    results.avg_track_continuity = total_track_continuity / num_comp_tracks;
  }

  // -------------------------------------------------------------------------
  // Track purity: fraction of track dominated by a single GT/computed match

  // For each computed track, find the GT track it matches most often
  std::map< int, std::map< int, int > > comp_to_gt_matches;
  std::map< int, std::map< int, int > > gt_to_comp_matches;
  std::map< int, int > comp_track_total_matches;
  std::map< int, int > gt_track_total_matches;

  for( int frame_id : frame_list )
  {
    const auto& fm = m_frame_matches[frame_id];

    auto comp_it = computed_by_frame.find( frame_id );
    auto gt_it = gt_by_frame.find( frame_id );

    std::vector< detection > frame_computed;
    std::vector< detection > frame_gt;

    if( comp_it != computed_by_frame.end() )
    {
      for( size_t idx : comp_it->second )
      {
        frame_computed.push_back( m_computed[idx] );
      }
    }
    if( gt_it != gt_by_frame.end() )
    {
      for( size_t idx : gt_it->second )
      {
        frame_gt.push_back( m_groundtruth[idx] );
      }
    }

    for( const auto& m : fm.matches )
    {
      if( m.computed_idx < 0 || m.gt_idx < 0 )
      {
        continue;
      }
      if( static_cast< size_t >( m.computed_idx ) >= frame_computed.size() ||
          static_cast< size_t >( m.gt_idx ) >= frame_gt.size() )
      {
        continue;
      }

      int gt_track = frame_gt[m.gt_idx].track_id;
      int comp_track = frame_computed[m.computed_idx].track_id;

      if( gt_track >= 0 && comp_track >= 0 )
      {
        comp_to_gt_matches[comp_track][gt_track]++;
        gt_to_comp_matches[gt_track][comp_track]++;
        comp_track_total_matches[comp_track]++;
        gt_track_total_matches[gt_track]++;
      }
    }
  }

  // Compute track purity (computed tracks)
  double total_track_purity = 0;
  int purity_track_count = 0;
  for( const auto& p : comp_to_gt_matches )
  {
    int total = comp_track_total_matches[p.first];
    if( total > 0 )
    {
      int max_matches = 0;
      for( const auto& gt_count : p.second )
      {
        max_matches = std::max( max_matches, gt_count.second );
      }
      total_track_purity += static_cast< double >( max_matches ) / total;
      purity_track_count++;
    }
  }

  if( purity_track_count > 0 )
  {
    results.avg_track_purity = total_track_purity / purity_track_count;
  }

  // Compute target purity (GT tracks)
  double total_target_purity = 0;
  int target_purity_count = 0;
  for( const auto& p : gt_to_comp_matches )
  {
    int total = gt_track_total_matches[p.first];
    if( total > 0 )
    {
      int max_matches = 0;
      for( const auto& comp_count : p.second )
      {
        max_matches = std::max( max_matches, comp_count.second );
      }
      total_target_purity += static_cast< double >( max_matches ) / total;
      target_purity_count++;
    }
  }

  if( target_purity_count > 0 )
  {
    results.avg_target_purity = total_target_purity / target_purity_count;
  }

  // -------------------------------------------------------------------------
  // Track Pd and FA

  // Pd = fraction of GT tracks that were matched at least once
  int matched_gt_tracks = 0;
  for( const auto& p : gt_track_matched_per_frame )
  {
    bool ever_matched = false;
    for( bool m : p.second )
    {
      if( m )
      {
        ever_matched = true;
        break;
      }
    }
    if( ever_matched )
    {
      matched_gt_tracks++;
    }
  }

  if( !all_gt_tracks.empty() )
  {
    results.track_pd = static_cast< double >( matched_gt_tracks ) / all_gt_tracks.size();
  }

  // FA = fraction of computed tracks that never matched any GT
  int unmatched_comp_tracks = 0;
  for( const auto& p : comp_track_matched_per_frame )
  {
    bool ever_matched = false;
    for( bool m : p.second )
    {
      if( m )
      {
        ever_matched = true;
        break;
      }
    }
    if( !ever_matched )
    {
      unmatched_comp_tracks++;
    }
  }

  if( !all_comp_tracks.empty() )
  {
    results.track_fa = static_cast< double >( unmatched_comp_tracks ) / all_comp_tracks.size();
  }
}

// =============================================================================
// Track quality metrics computation

void
model_evaluator::priv::compute_track_quality_metrics( evaluation_results& results )
{
  if( !m_config.compute_tracking_metrics )
  {
    return;
  }

  // Compute track lengths
  std::map< int, int > gt_track_lengths;
  std::map< int, int > comp_track_lengths;

  for( const auto& d : m_groundtruth )
  {
    if( d.track_id >= 0 )
    {
      gt_track_lengths[d.track_id]++;
    }
  }

  for( const auto& d : m_computed )
  {
    if( d.track_id >= 0 )
    {
      comp_track_lengths[d.track_id]++;
    }
  }

  // Average GT track length
  if( !gt_track_lengths.empty() )
  {
    double sum = 0;
    for( const auto& p : gt_track_lengths )
    {
      sum += p.second;
    }
    results.avg_gt_track_length = sum / gt_track_lengths.size();
  }

  // Average computed track length
  if( !comp_track_lengths.empty() )
  {
    double sum = 0;
    for( const auto& p : comp_track_lengths )
    {
      sum += p.second;
    }
    results.avg_track_length = sum / comp_track_lengths.size();
  }

  // Track completeness: for each GT track, what fraction is covered by
  // the best matching computed track
  // Use cached frame groupings
  const auto& computed_by_frame = m_computed_by_frame;
  const auto& gt_by_frame = m_gt_by_frame;

  std::vector< int > frame_list;
  for( const auto& fm_pair : m_frame_matches )
  {
    frame_list.push_back( fm_pair.first );
  }
  std::sort( frame_list.begin(), frame_list.end() );

  // Count matches per (gt_track, comp_track) pair
  std::map< int, std::map< int, int > > gt_to_comp_matches;

  for( int frame_id : frame_list )
  {
    const auto& fm = m_frame_matches.at( frame_id );

    auto comp_it = computed_by_frame.find( frame_id );
    auto gt_it = gt_by_frame.find( frame_id );

    std::vector< detection > frame_computed;
    std::vector< detection > frame_gt;

    if( comp_it != computed_by_frame.end() )
    {
      for( size_t idx : comp_it->second )
      {
        frame_computed.push_back( m_computed[idx] );
      }
    }
    if( gt_it != gt_by_frame.end() )
    {
      for( size_t idx : gt_it->second )
      {
        frame_gt.push_back( m_groundtruth[idx] );
      }
    }

    for( const auto& m : fm.matches )
    {
      if( m.computed_idx < 0 || m.gt_idx < 0 )
      {
        continue;
      }
      if( static_cast< size_t >( m.computed_idx ) >= frame_computed.size() ||
          static_cast< size_t >( m.gt_idx ) >= frame_gt.size() )
      {
        continue;
      }

      int gt_track = frame_gt[m.gt_idx].track_id;
      int comp_track = frame_computed[m.computed_idx].track_id;

      if( gt_track >= 0 && comp_track >= 0 )
      {
        gt_to_comp_matches[gt_track][comp_track]++;
      }
    }
  }

  // Compute completeness for each GT track
  double total_completeness = 0;
  int num_gt_tracks = 0;

  for( const auto& p : gt_track_lengths )
  {
    int gt_track = p.first;
    int gt_len = p.second;

    auto match_it = gt_to_comp_matches.find( gt_track );
    if( match_it != gt_to_comp_matches.end() )
    {
      // Find best matching computed track
      int best_matches = 0;
      for( const auto& comp_matches : match_it->second )
      {
        best_matches = std::max( best_matches, comp_matches.second );
      }
      total_completeness += static_cast< double >( best_matches ) / gt_len;
    }
    // Else: no matches, contributes 0 to completeness

    num_gt_tracks++;
  }

  if( num_gt_tracks > 0 )
  {
    results.track_completeness = total_completeness / num_gt_tracks;
  }

  // Average gap length: compute gaps in GT track coverage
  std::map< int, std::vector< bool > > gt_track_matched_per_frame;

  for( int frame_id : frame_list )
  {
    const auto& fm = m_frame_matches.at( frame_id );

    auto gt_it = gt_by_frame.find( frame_id );
    if( gt_it == gt_by_frame.end() )
    {
      continue;
    }

    std::vector< detection > frame_gt;
    for( size_t idx : gt_it->second )
    {
      frame_gt.push_back( m_groundtruth[idx] );
    }

    std::set< int > matched_gt_tracks;
    auto comp_it = computed_by_frame.find( frame_id );
    std::vector< detection > frame_computed;
    if( comp_it != computed_by_frame.end() )
    {
      for( size_t idx : comp_it->second )
      {
        frame_computed.push_back( m_computed[idx] );
      }
    }

    for( const auto& m : fm.matches )
    {
      if( m.gt_idx >= 0 && static_cast< size_t >( m.gt_idx ) < frame_gt.size() )
      {
        matched_gt_tracks.insert( frame_gt[m.gt_idx].track_id );
      }
    }

    for( const auto& d : frame_gt )
    {
      if( d.track_id >= 0 )
      {
        gt_track_matched_per_frame[d.track_id].push_back(
          matched_gt_tracks.count( d.track_id ) > 0 );
      }
    }
  }

  // Count gaps and their lengths
  double total_gap_length = 0;
  int num_gaps = 0;

  for( const auto& p : gt_track_matched_per_frame )
  {
    const auto& matched_status = p.second;
    int current_gap = 0;
    bool in_track = false;

    for( bool matched : matched_status )
    {
      if( matched )
      {
        if( in_track && current_gap > 0 )
        {
          // End of a gap
          total_gap_length += current_gap;
          num_gaps++;
        }
        current_gap = 0;
        in_track = true;
      }
      else if( in_track )
      {
        // We're in a gap
        current_gap++;
      }
    }
    // Don't count trailing gaps (track ended)
  }

  if( num_gaps > 0 )
  {
    results.avg_gap_length = total_gap_length / num_gaps;
  }
}

// =============================================================================
// Average Precision computation

void
model_evaluator::priv::compute_average_precision( evaluation_results& results )
{
  // Collect all detections with their match status
  struct scored_detection
  {
    double confidence;
    bool is_tp;
  };

  std::vector< scored_detection > all_detections;

  // Use cached frame groupings
  const auto& computed_by_frame = m_computed_by_frame;

  for( const auto& fm_pair : m_frame_matches )
  {
    int frame_id = fm_pair.first;
    const auto& fm = fm_pair.second;

    auto comp_it = computed_by_frame.find( frame_id );
    if( comp_it == computed_by_frame.end() )
    {
      continue;
    }

    std::vector< detection > frame_computed;
    for( size_t idx : comp_it->second )
    {
      frame_computed.push_back( m_computed[idx] );
    }

    // Mark which detections are TPs
    std::set< int > tp_indices;
    for( const auto& m : fm.matches )
    {
      tp_indices.insert( m.computed_idx );
    }

    // Add all detections
    for( size_t i = 0; i < frame_computed.size(); i++ )
    {
      scored_detection sd;
      sd.confidence = frame_computed[i].confidence;
      sd.is_tp = tp_indices.count( static_cast< int >( i ) ) > 0;
      all_detections.push_back( sd );
    }
  }

  if( all_detections.empty() )
  {
    return;
  }

  // Sort by confidence descending
  std::sort( all_detections.begin(), all_detections.end(),
    []( const scored_detection& a, const scored_detection& b )
    {
      return a.confidence > b.confidence;
    } );

  // Compute precision-recall curve
  double total_positives = results.true_positives + results.false_negatives;
  if( total_positives <= 0 )
  {
    return;
  }

  std::vector< double > precisions;
  std::vector< double > recalls;

  double tp = 0, fp = 0;
  for( const auto& sd : all_detections )
  {
    if( sd.is_tp )
    {
      tp++;
    }
    else
    {
      fp++;
    }

    double precision = tp / ( tp + fp );
    double recall = tp / total_positives;

    precisions.push_back( precision );
    recalls.push_back( recall );
  }

  // Compute AP using 11-point interpolation or all-point interpolation
  // Using all-point interpolation (area under PR curve)

  // Make precision monotonically decreasing (from right to left)
  std::vector< double > interp_precisions = precisions;
  for( int i = static_cast< int >( interp_precisions.size() ) - 2; i >= 0; i-- )
  {
    interp_precisions[i] = std::max( interp_precisions[i], interp_precisions[i + 1] );
  }

  // Compute area under curve
  double ap = 0;
  double prev_recall = 0;
  for( size_t i = 0; i < recalls.size(); i++ )
  {
    double recall_diff = recalls[i] - prev_recall;
    ap += interp_precisions[i] * recall_diff;
    prev_recall = recalls[i];
  }

  results.average_precision = ap;
}

// =============================================================================
// Multi-threshold AP computation (COCO-style)

void
model_evaluator::priv::compute_multi_threshold_ap( evaluation_results& results )
{
  // Compute AP at specific IoU thresholds using cached IoU matrices

  auto compute_ap_at_threshold = [this]( double iou_thresh ) -> double
  {
    // Collect all detections with TP/FP status at this threshold
    struct scored_det
    {
      double conf;
      bool is_tp;
    };
    std::vector< scored_det > all_dets;
    all_dets.reserve( m_computed.size() );
    double total_gt = m_groundtruth.size();

    for( int frame_id : m_frame_list )
    {
      auto comp_it = m_computed_by_frame.find( frame_id );
      auto gt_it = m_gt_by_frame.find( frame_id );
      auto iou_it = m_iou_matrices.find( frame_id );

      if( comp_it == m_computed_by_frame.end() )
      {
        continue;
      }

      const auto& comp_indices = comp_it->second;
      size_t num_gt = ( gt_it != m_gt_by_frame.end() ) ? gt_it->second.size() : 0;

      // Sort computed by confidence descending
      std::vector< size_t > sorted_local( comp_indices.size() );
      std::iota( sorted_local.begin(), sorted_local.end(), 0 );
      std::sort( sorted_local.begin(), sorted_local.end(),
        [this, &comp_indices]( size_t a, size_t b )
        {
          return m_computed[comp_indices[a]].confidence >
                 m_computed[comp_indices[b]].confidence;
        } );

      std::vector< bool > gt_matched( num_gt, false );

      for( size_t ci : sorted_local )
      {
        double best_iou = 0;
        int best_gt = -1;

        if( iou_it != m_iou_matrices.end() && num_gt > 0 )
        {
          const auto& iou_matrix = iou_it->second;
          for( size_t gi = 0; gi < num_gt; gi++ )
          {
            if( !gt_matched[gi] && iou_matrix[ci][gi] > best_iou )
            {
              best_iou = iou_matrix[ci][gi];
              best_gt = static_cast< int >( gi );
            }
          }
        }

        scored_det sd;
        sd.conf = m_computed[comp_indices[ci]].confidence;

        if( best_gt >= 0 && best_iou >= iou_thresh )
        {
          sd.is_tp = true;
          gt_matched[best_gt] = true;
        }
        else
        {
          sd.is_tp = false;
        }

        all_dets.push_back( sd );
      }
    }

    if( total_gt <= 0 || all_dets.empty() )
    {
      return 0.0;
    }

    // Sort all detections by confidence
    std::sort( all_dets.begin(), all_dets.end(),
      []( const scored_det& a, const scored_det& b )
      {
        return a.conf > b.conf;
      } );

    // Compute PR curve
    std::vector< double > precisions, recalls;
    precisions.reserve( all_dets.size() );
    recalls.reserve( all_dets.size() );
    double tp = 0, fp = 0;

    for( const auto& sd : all_dets )
    {
      if( sd.is_tp )
      {
        tp++;
      }
      else
      {
        fp++;
      }
      precisions.push_back( tp / ( tp + fp ) );
      recalls.push_back( tp / total_gt );
    }

    // Make precision monotonically decreasing
    for( int i = static_cast< int >( precisions.size() ) - 2; i >= 0; i-- )
    {
      precisions[i] = std::max( precisions[i], precisions[i + 1] );
    }

    // Compute AP
    double ap = 0;
    double prev_recall = 0;
    for( size_t i = 0; i < recalls.size(); i++ )
    {
      ap += precisions[i] * ( recalls[i] - prev_recall );
      prev_recall = recalls[i];
    }

    return ap;
  };

  // AP@0.5
  results.ap50 = compute_ap_at_threshold( 0.5 );

  // AP@0.75
  results.ap75 = compute_ap_at_threshold( 0.75 );

  // AP@0.5:0.95 (average over 10 thresholds)
  // Use parallel computation for multiple thresholds
  std::vector< double > thresholds;
  for( double thresh = 0.5; thresh <= 0.95; thresh += 0.05 )
  {
    thresholds.push_back( thresh );
  }

  double sum_ap = 0;

  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:sum_ap)
  #endif
  for( size_t i = 0; i < thresholds.size(); i++ )
  {
    sum_ap += compute_ap_at_threshold( thresholds[i] );
  }

  if( !thresholds.empty() )
  {
    results.ap50_95 = sum_ap / thresholds.size();
  }
}

// =============================================================================
// Classification metrics computation

void
model_evaluator::priv::compute_classification_metrics( evaluation_results& results )
{
  // Classification accuracy: among TPs, how many have correct class
  int correct_class = 0;
  int total_matches = 0;

  // Use cached frame groupings
  const auto& computed_by_frame = m_computed_by_frame;
  const auto& gt_by_frame = m_gt_by_frame;

  for( const auto& fm_pair : m_frame_matches )
  {
    int frame_id = fm_pair.first;
    const auto& fm = fm_pair.second;

    auto comp_it = computed_by_frame.find( frame_id );
    auto gt_it = gt_by_frame.find( frame_id );

    std::vector< detection > frame_computed;
    std::vector< detection > frame_gt;

    if( comp_it != computed_by_frame.end() )
    {
      for( size_t idx : comp_it->second )
      {
        frame_computed.push_back( m_computed[idx] );
      }
    }
    if( gt_it != gt_by_frame.end() )
    {
      for( size_t idx : gt_it->second )
      {
        frame_gt.push_back( m_groundtruth[idx] );
      }
    }

    for( const auto& m : fm.matches )
    {
      if( m.computed_idx < 0 || m.gt_idx < 0 )
      {
        continue;
      }
      if( static_cast< size_t >( m.computed_idx ) >= frame_computed.size() ||
          static_cast< size_t >( m.gt_idx ) >= frame_gt.size() )
      {
        continue;
      }

      const auto& comp = frame_computed[m.computed_idx];
      const auto& gt = frame_gt[m.gt_idx];

      if( comp.class_name == gt.class_name )
      {
        correct_class++;
      }
      total_matches++;
    }
  }

  if( total_matches > 0 )
  {
    results.classification_accuracy =
      static_cast< double >( correct_class ) / total_matches;
  }

  // Mean AP: computed from per-class metrics if enabled
  if( m_config.compute_per_class_metrics && !results.per_class_metrics.empty() )
  {
    double sum_ap = 0;
    int num_classes = 0;

    for( const auto& class_metrics : results.per_class_metrics )
    {
      // Use F1 as proxy for AP if not computing full AP per class
      auto f1_it = class_metrics.second.find( "f1_score" );
      if( f1_it != class_metrics.second.end() && f1_it->second > 0 )
      {
        sum_ap += f1_it->second;
        num_classes++;
      }
    }

    if( num_classes > 0 )
    {
      results.mean_ap = sum_ap / num_classes;
    }
  }
}

// =============================================================================
// Per-class metrics computation

void
model_evaluator::priv::compute_per_class_metrics( evaluation_results& results )
{
  if( !m_config.compute_per_class_metrics )
  {
    return;
  }

  // Get all unique classes
  std::set< std::string > all_classes;
  for( const auto& d : m_groundtruth )
  {
    all_classes.insert( d.class_name );
  }
  for( const auto& d : m_computed )
  {
    all_classes.insert( d.class_name );
  }

  // Group detections by class
  std::map< std::string, std::vector< detection > > gt_by_class;
  std::map< std::string, std::vector< detection > > comp_by_class;

  for( const auto& d : m_groundtruth )
  {
    gt_by_class[d.class_name].push_back( d );
  }
  for( const auto& d : m_computed )
  {
    comp_by_class[d.class_name].push_back( d );
  }

  // Compute metrics for each class
  for( const auto& class_name : all_classes )
  {
    const auto& class_gt = gt_by_class[class_name];
    const auto& class_comp = comp_by_class[class_name];

    // Create temporary evaluator for this class
    model_evaluator class_eval;
    evaluation_config class_config = m_config;
    class_config.compute_per_class_metrics = false;
    class_config.compute_tracking_metrics = false;
    class_eval.set_config( class_config );

    // We need to do per-frame matching for this class only
    // Group by frame
    std::map< int, std::vector< detection > > class_gt_by_frame;
    std::map< int, std::vector< detection > > class_comp_by_frame;

    for( const auto& d : class_gt )
    {
      class_gt_by_frame[d.frame_id].push_back( d );
    }
    for( const auto& d : class_comp )
    {
      class_comp_by_frame[d.frame_id].push_back( d );
    }

    std::set< int > class_frames;
    for( const auto& p : class_gt_by_frame )
    {
      class_frames.insert( p.first );
    }
    for( const auto& p : class_comp_by_frame )
    {
      class_frames.insert( p.first );
    }

    double tp = 0, fp = 0, fn = 0;

    for( int frame_id : class_frames )
    {
      const auto& frame_gt = class_gt_by_frame[frame_id];
      const auto& frame_comp = class_comp_by_frame[frame_id];

      frame_matches fm;
      match_frame( frame_comp, frame_gt, fm );

      tp += fm.matches.size();
      fp += fm.false_positives.size();
      fn += fm.false_negatives.size();
    }

    std::map< std::string, double > class_metrics;
    class_metrics["true_positives"] = tp;
    class_metrics["false_positives"] = fp;
    class_metrics["false_negatives"] = fn;

    if( tp + fp > 0 )
    {
      class_metrics["precision"] = tp / ( tp + fp );
    }
    else
    {
      class_metrics["precision"] = 0;
    }

    if( tp + fn > 0 )
    {
      class_metrics["recall"] = tp / ( tp + fn );
    }
    else
    {
      class_metrics["recall"] = 0;
    }

    double prec = class_metrics["precision"];
    double rec = class_metrics["recall"];
    if( prec + rec > 0 )
    {
      class_metrics["f1_score"] = 2.0 * prec * rec / ( prec + rec );
    }
    else
    {
      class_metrics["f1_score"] = 0;
    }

    class_metrics["total_gt"] = class_gt.size();
    class_metrics["total_computed"] = class_comp.size();

    results.per_class_metrics[class_name] = class_metrics;
  }
}

// =============================================================================
// evaluation_results implementation

void
evaluation_results::populate_all_metrics()
{
  all_metrics.clear();

  // Detection metrics
  all_metrics["true_positives"] = true_positives;
  all_metrics["false_positives"] = false_positives;
  all_metrics["false_negatives"] = false_negatives;
  all_metrics["precision"] = precision;
  all_metrics["recall"] = recall;
  all_metrics["f1_score"] = f1_score;
  all_metrics["mcc"] = mcc;
  all_metrics["average_precision"] = average_precision;
  all_metrics["ap50"] = ap50;
  all_metrics["ap75"] = ap75;
  all_metrics["ap50_95"] = ap50_95;
  all_metrics["false_discovery_rate"] = false_discovery_rate;
  all_metrics["miss_rate"] = miss_rate;

  // Localization metrics
  all_metrics["mean_iou"] = mean_iou;
  all_metrics["median_iou"] = median_iou;
  all_metrics["mean_center_distance"] = mean_center_distance;
  all_metrics["mean_size_error"] = mean_size_error;

  // MOT metrics
  all_metrics["mota"] = mota;
  all_metrics["motp"] = motp;
  all_metrics["idf1"] = idf1;
  all_metrics["idp"] = idp;
  all_metrics["idr"] = idr;
  all_metrics["id_switches"] = id_switches;
  all_metrics["fragmentations"] = fragmentations;
  all_metrics["mostly_tracked"] = mostly_tracked;
  all_metrics["partially_tracked"] = partially_tracked;
  all_metrics["mostly_lost"] = mostly_lost;
  all_metrics["num_transfer"] = num_transfer;
  all_metrics["num_ascend"] = num_ascend;
  all_metrics["num_migrate"] = num_migrate;

  // HOTA metrics
  all_metrics["hota"] = hota;
  all_metrics["deta"] = deta;
  all_metrics["assa"] = assa;
  all_metrics["loca"] = loca;

  // KWANT metrics
  all_metrics["avg_track_continuity"] = avg_track_continuity;
  all_metrics["avg_track_purity"] = avg_track_purity;
  all_metrics["avg_target_continuity"] = avg_target_continuity;
  all_metrics["avg_target_purity"] = avg_target_purity;
  all_metrics["track_pd"] = track_pd;
  all_metrics["track_fa"] = track_fa;

  // Track quality metrics
  all_metrics["avg_track_length"] = avg_track_length;
  all_metrics["avg_gt_track_length"] = avg_gt_track_length;
  all_metrics["track_completeness"] = track_completeness;
  all_metrics["avg_gap_length"] = avg_gap_length;

  // Normalized rates
  all_metrics["mt_ratio"] = mt_ratio;
  all_metrics["pt_ratio"] = pt_ratio;
  all_metrics["ml_ratio"] = ml_ratio;
  all_metrics["faf"] = faf;

  // Classification metrics
  all_metrics["classification_accuracy"] = classification_accuracy;
  all_metrics["mean_ap"] = mean_ap;

  // Statistics
  all_metrics["total_gt_objects"] = total_gt_objects;
  all_metrics["total_computed"] = total_computed;
  all_metrics["total_frames"] = total_frames;
  all_metrics["total_gt_tracks"] = total_gt_tracks;
  all_metrics["total_computed_tracks"] = total_computed_tracks;
}

// =============================================================================
// model_evaluator implementation

model_evaluator::model_evaluator()
  : d( new priv )
{
}

model_evaluator::~model_evaluator() = default;

model_evaluator::model_evaluator( model_evaluator&& ) noexcept = default;

model_evaluator&
model_evaluator::operator=( model_evaluator&& ) noexcept = default;

void
model_evaluator::set_config( const evaluation_config& config )
{
  d->m_config = config;
}

evaluation_config
model_evaluator::get_config() const
{
  return d->m_config;
}

evaluation_results
model_evaluator::evaluate(
  const std::vector< std::string >& computed_files,
  const std::vector< std::string >& groundtruth_files )
{
  evaluation_results results;

  if( computed_files.size() != groundtruth_files.size() )
  {
    LOG_ERROR( d->m_logger,
      "Mismatch between computed files (" << computed_files.size()
      << ") and groundtruth files (" << groundtruth_files.size() << ")" );
    results.populate_all_metrics();
    return results;
  }

  // Clear previous data
  d->m_computed.clear();
  d->m_groundtruth.clear();

  // Load all files
  for( size_t i = 0; i < computed_files.size(); i++ )
  {
    if( !d->parse_viame_csv( computed_files[i], d->m_computed ) )
    {
      LOG_WARN( d->m_logger, "Failed to parse computed file: " << computed_files[i] );
    }
    if( !d->parse_viame_csv( groundtruth_files[i], d->m_groundtruth ) )
    {
      LOG_WARN( d->m_logger, "Failed to parse groundtruth file: " << groundtruth_files[i] );
    }
  }

  LOG_INFO( d->m_logger, "Loaded " << d->m_computed.size()
            << " computed detections and " << d->m_groundtruth.size()
            << " ground truth annotations" );

  // Build caches for efficient processing
  d->build_caches();

  // Perform matching using cached IoU matrices
  d->perform_matching();

  // Compute all metrics
  d->compute_detection_metrics( results );
  d->compute_localization_metrics( results );
  d->compute_mot_metrics( results );
  d->compute_hota_metrics( results );
  d->compute_kwant_metrics( results );
  d->compute_track_quality_metrics( results );
  d->compute_average_precision( results );
  d->compute_multi_threshold_ap( results );
  d->compute_per_class_metrics( results );
  d->compute_classification_metrics( results );

  // Populate the combined map
  results.populate_all_metrics();

  LOG_INFO( d->m_logger, "Evaluation complete: "
            << "Precision=" << std::fixed << std::setprecision( 4 ) << results.precision
            << ", Recall=" << results.recall
            << ", F1=" << results.f1_score
            << ", MOTA=" << results.mota
            << ", IDF1=" << results.idf1
            << ", HOTA=" << results.hota );

  return results;
}

std::map< std::string, double >
model_evaluator::evaluate_to_map(
  const std::vector< std::string >& computed_files,
  const std::vector< std::string >& groundtruth_files )
{
  auto results = evaluate( computed_files, groundtruth_files );
  return results.all_metrics;
}

// =============================================================================
// Convenience function

std::map< std::string, double >
evaluate_models(
  const std::vector< std::string >& computed_files,
  const std::vector< std::string >& groundtruth_files,
  const evaluation_config& config )
{
  model_evaluator evaluator;
  evaluator.set_config( config );
  return evaluator.evaluate_to_map( computed_files, groundtruth_files );
}

// =============================================================================
// Plotting data generation
// =============================================================================

pr_curve_data
model_evaluator::generate_pr_curve( int num_points )
{
  pr_curve_data result;

  if( d->m_computed.empty() || d->m_groundtruth.empty() )
  {
    return result;
  }

  // Sort computed detections by confidence descending
  std::vector< size_t > sorted_indices( d->m_computed.size() );
  std::iota( sorted_indices.begin(), sorted_indices.end(), 0 );
  std::sort( sorted_indices.begin(), sorted_indices.end(),
    [this]( size_t a, size_t b )
    {
      return d->m_computed[a].confidence > d->m_computed[b].confidence;
    } );

  const int total_gt = static_cast< int >( d->m_groundtruth.size() );
  int tp = 0, fp = 0;

  // Track which GT have been matched
  std::vector< bool > gt_matched( d->m_groundtruth.size(), false );

  // Group GT by frame for efficient lookup
  std::map< int, std::vector< size_t > > gt_by_frame;
  for( size_t i = 0; i < d->m_groundtruth.size(); i++ )
  {
    gt_by_frame[d->m_groundtruth[i].frame_id].push_back( i );
  }

  // Process detections in confidence order
  std::vector< std::tuple< double, int, int > > curve_points;  // (conf, tp, fp)
  double prev_conf = std::numeric_limits< double >::max();

  for( size_t idx : sorted_indices )
  {
    const auto& det = d->m_computed[idx];
    double conf = det.confidence;

    // Check if this detection matches any unmatched GT
    bool matched = false;
    auto frame_it = gt_by_frame.find( det.frame_id );
    if( frame_it != gt_by_frame.end() )
    {
      double best_iou = 0.0;
      size_t best_gt = 0;

      for( size_t gi : frame_it->second )
      {
        if( gt_matched[gi] )
          continue;

        double iou = d->compute_iou( det, d->m_groundtruth[gi] );
        if( iou > best_iou && iou >= d->m_config.iou_threshold )
        {
          best_iou = iou;
          best_gt = gi;
          matched = true;
        }
      }

      if( matched )
      {
        gt_matched[best_gt] = true;
        tp++;
      }
      else
      {
        fp++;
      }
    }
    else
    {
      fp++;
    }

    // Record point when confidence changes
    if( conf != prev_conf )
    {
      curve_points.push_back( { conf, tp, fp } );
      prev_conf = conf;
    }
  }

  // Add final point
  curve_points.push_back( { 0.0, tp, fp } );

  // Convert to PR curve points
  result.points.reserve( curve_points.size() );
  double max_f1 = 0.0;
  double best_threshold = 0.0;

  for( const auto& pt : curve_points )
  {
    pr_curve_point prp;
    prp.confidence = std::get<0>( pt );
    prp.tp = std::get<1>( pt );
    prp.fp = std::get<2>( pt );
    prp.fn = total_gt - prp.tp;

    if( prp.tp + prp.fp > 0 )
    {
      prp.precision = static_cast< double >( prp.tp ) / ( prp.tp + prp.fp );
    }
    if( total_gt > 0 )
    {
      prp.recall = static_cast< double >( prp.tp ) / total_gt;
    }
    if( prp.precision + prp.recall > 0 )
    {
      prp.f1 = 2.0 * prp.precision * prp.recall / ( prp.precision + prp.recall );
    }

    if( prp.f1 > max_f1 )
    {
      max_f1 = prp.f1;
      best_threshold = prp.confidence;
    }

    result.points.push_back( prp );
  }

  result.max_f1 = max_f1;
  result.best_threshold = best_threshold;

  // Compute Average Precision using 11-point interpolation
  // (PASCAL VOC style)
  double ap = 0.0;
  for( double r = 0.0; r <= 1.0; r += 0.1 )
  {
    double max_prec = 0.0;
    for( const auto& pt : result.points )
    {
      if( pt.recall >= r )
      {
        max_prec = std::max( max_prec, pt.precision );
      }
    }
    ap += max_prec;
  }
  result.average_precision = ap / 11.0;

  return result;
}

std::map< std::string, pr_curve_data >
model_evaluator::generate_per_class_pr_curves( int num_points )
{
  std::map< std::string, pr_curve_data > result;

  // Get all class names
  std::set< std::string > all_classes;
  for( const auto& det : d->m_computed )
  {
    if( !det.class_name.empty() )
    {
      all_classes.insert( det.class_name );
    }
  }
  for( const auto& det : d->m_groundtruth )
  {
    if( !det.class_name.empty() )
    {
      all_classes.insert( det.class_name );
    }
  }

  // Generate PR curve for each class
  for( const auto& class_name : all_classes )
  {
    // Filter detections for this class
    std::vector< detection > class_computed;
    std::vector< detection > class_gt;

    for( const auto& det : d->m_computed )
    {
      if( det.class_name == class_name )
      {
        class_computed.push_back( det );
      }
    }
    for( const auto& det : d->m_groundtruth )
    {
      if( det.class_name == class_name )
      {
        class_gt.push_back( det );
      }
    }

    if( class_gt.empty() )
      continue;

    // Sort by confidence
    std::sort( class_computed.begin(), class_computed.end(),
      []( const detection& a, const detection& b )
      {
        return a.confidence > b.confidence;
      } );

    pr_curve_data curve;
    curve.class_name = class_name;

    const int total_gt = static_cast< int >( class_gt.size() );
    int tp = 0, fp = 0;

    std::vector< bool > gt_matched( class_gt.size(), false );

    // Group GT by frame
    std::map< int, std::vector< size_t > > gt_by_frame;
    for( size_t i = 0; i < class_gt.size(); i++ )
    {
      gt_by_frame[class_gt[i].frame_id].push_back( i );
    }

    double prev_conf = std::numeric_limits< double >::max();

    for( const auto& det : class_computed )
    {
      bool matched = false;
      auto frame_it = gt_by_frame.find( det.frame_id );
      if( frame_it != gt_by_frame.end() )
      {
        double best_iou = 0.0;
        size_t best_gt = 0;

        for( size_t gi : frame_it->second )
        {
          if( gt_matched[gi] )
            continue;

          double iou = d->compute_iou( det, class_gt[gi] );
          if( iou > best_iou && iou >= d->m_config.iou_threshold )
          {
            best_iou = iou;
            best_gt = gi;
            matched = true;
          }
        }

        if( matched )
        {
          gt_matched[best_gt] = true;
          tp++;
        }
        else
        {
          fp++;
        }
      }
      else
      {
        fp++;
      }

      if( det.confidence != prev_conf )
      {
        pr_curve_point prp;
        prp.confidence = det.confidence;
        prp.tp = tp;
        prp.fp = fp;
        prp.fn = total_gt - tp;

        if( tp + fp > 0 )
        {
          prp.precision = static_cast< double >( tp ) / ( tp + fp );
        }
        if( total_gt > 0 )
        {
          prp.recall = static_cast< double >( tp ) / total_gt;
        }
        if( prp.precision + prp.recall > 0 )
        {
          prp.f1 = 2.0 * prp.precision * prp.recall / ( prp.precision + prp.recall );
        }

        curve.points.push_back( prp );
        prev_conf = det.confidence;
      }
    }

    // Final point
    pr_curve_point final_pt;
    final_pt.confidence = 0.0;
    final_pt.tp = tp;
    final_pt.fp = fp;
    final_pt.fn = total_gt - tp;
    if( tp + fp > 0 )
    {
      final_pt.precision = static_cast< double >( tp ) / ( tp + fp );
    }
    if( total_gt > 0 )
    {
      final_pt.recall = static_cast< double >( tp ) / total_gt;
    }
    if( final_pt.precision + final_pt.recall > 0 )
    {
      final_pt.f1 = 2.0 * final_pt.precision * final_pt.recall /
                    ( final_pt.precision + final_pt.recall );
    }
    curve.points.push_back( final_pt );

    // Compute max F1 and AP
    double max_f1 = 0.0;
    for( const auto& pt : curve.points )
    {
      if( pt.f1 > max_f1 )
      {
        max_f1 = pt.f1;
        curve.best_threshold = pt.confidence;
      }
    }
    curve.max_f1 = max_f1;

    // 11-point AP
    double ap = 0.0;
    for( double r = 0.0; r <= 1.0; r += 0.1 )
    {
      double max_prec = 0.0;
      for( const auto& pt : curve.points )
      {
        if( pt.recall >= r )
        {
          max_prec = std::max( max_prec, pt.precision );
        }
      }
      ap += max_prec;
    }
    curve.average_precision = ap / 11.0;

    result[class_name] = curve;
  }

  return result;
}

confusion_matrix_data
model_evaluator::generate_confusion_matrix()
{
  confusion_matrix_data result;

  // Get all class names and assign indices
  std::set< std::string > all_classes;
  for( const auto& det : d->m_computed )
  {
    if( !det.class_name.empty() )
    {
      all_classes.insert( det.class_name );
    }
  }
  for( const auto& det : d->m_groundtruth )
  {
    if( !det.class_name.empty() )
    {
      all_classes.insert( det.class_name );
    }
  }

  // Convert to vector and add "background" class for FP/FN
  result.class_names = std::vector< std::string >( all_classes.begin(), all_classes.end() );
  std::sort( result.class_names.begin(), result.class_names.end() );
  result.class_names.push_back( "background" );

  std::map< std::string, int > class_to_idx;
  for( size_t i = 0; i < result.class_names.size(); i++ )
  {
    class_to_idx[result.class_names[i]] = static_cast< int >( i );
  }

  int bg_idx = static_cast< int >( result.class_names.size() ) - 1;

  // Initialize matrix
  int n = static_cast< int >( result.class_names.size() );
  result.matrix.resize( n, std::vector< int >( n, 0 ) );

  // Use cached frame matches to build confusion matrix
  for( const auto& frame_match_pair : d->m_frame_matches )
  {
    const frame_matches& fm = frame_match_pair.second;
    int frame_id = frame_match_pair.first;

    // Get indices for this frame
    const auto& comp_indices = d->m_computed_by_frame.at( frame_id );
    const auto& gt_indices = d->m_gt_by_frame.count( frame_id ) ?
                             d->m_gt_by_frame.at( frame_id ) :
                             std::vector< size_t >();

    // True positives - matched pairs
    for( const auto& match : fm.matches )
    {
      size_t comp_global = comp_indices[match.computed_idx];
      size_t gt_global = gt_indices[match.gt_idx];

      std::string gt_class = d->m_groundtruth[gt_global].class_name;
      std::string pred_class = d->m_computed[comp_global].class_name;

      if( gt_class.empty() ) gt_class = "background";
      if( pred_class.empty() ) pred_class = "background";

      int gt_idx = class_to_idx.count( gt_class ) ? class_to_idx[gt_class] : bg_idx;
      int pred_idx = class_to_idx.count( pred_class ) ? class_to_idx[pred_class] : bg_idx;

      result.matrix[gt_idx][pred_idx]++;
    }

    // False positives - predicted but no GT match
    for( int fp_idx : fm.false_positives )
    {
      size_t comp_global = comp_indices[fp_idx];
      std::string pred_class = d->m_computed[comp_global].class_name;
      if( pred_class.empty() ) pred_class = "background";

      int pred_idx = class_to_idx.count( pred_class ) ? class_to_idx[pred_class] : bg_idx;
      result.matrix[bg_idx][pred_idx]++;  // GT is background
    }

    // False negatives - GT with no prediction match
    for( int fn_idx : fm.false_negatives )
    {
      size_t gt_global = gt_indices[fn_idx];
      std::string gt_class = d->m_groundtruth[gt_global].class_name;
      if( gt_class.empty() ) gt_class = "background";

      int gt_idx = class_to_idx.count( gt_class ) ? class_to_idx[gt_class] : bg_idx;
      result.matrix[gt_idx][bg_idx]++;  // Predicted as background
    }
  }

  // Compute normalized matrix
  result.normalized_matrix.resize( n, std::vector< double >( n, 0.0 ) );
  int total = 0;
  int correct = 0;

  for( int i = 0; i < n; i++ )
  {
    int row_sum = 0;
    for( int j = 0; j < n; j++ )
    {
      row_sum += result.matrix[i][j];
    }

    if( row_sum > 0 )
    {
      for( int j = 0; j < n; j++ )
      {
        result.normalized_matrix[i][j] =
          static_cast< double >( result.matrix[i][j] ) / row_sum;
      }
    }

    // Per-class accuracy (diagonal / row sum)
    if( row_sum > 0 && i < bg_idx )
    {
      result.per_class_accuracy[result.class_names[i]] =
        static_cast< double >( result.matrix[i][i] ) / row_sum;
    }

    total += row_sum;
    correct += result.matrix[i][i];
  }

  if( total > 0 )
  {
    result.overall_accuracy = static_cast< double >( correct ) / total;
  }

  return result;
}

roc_curve_data
model_evaluator::generate_roc_curve( int num_points )
{
  roc_curve_data result;

  if( d->m_computed.empty() || d->m_groundtruth.empty() )
  {
    return result;
  }

  // Sort computed detections by confidence descending
  std::vector< detection > sorted_computed = d->m_computed;
  std::sort( sorted_computed.begin(), sorted_computed.end(),
    []( const detection& a, const detection& b )
    {
      return a.confidence > b.confidence;
    } );

  const int total_gt = static_cast< int >( d->m_groundtruth.size() );
  const int total_frames = static_cast< int >( d->m_frame_list.size() );

  std::vector< bool > gt_matched( d->m_groundtruth.size(), false );

  int tp = 0, fp = 0;
  double prev_conf = std::numeric_limits< double >::max();

  for( const auto& det : sorted_computed )
  {
    bool matched = false;
    auto frame_it = d->m_gt_by_frame.find( det.frame_id );
    if( frame_it != d->m_gt_by_frame.end() )
    {
      double best_iou = 0.0;
      size_t best_gt = 0;

      for( size_t gi : frame_it->second )
      {
        if( gt_matched[gi] )
          continue;

        double iou = d->compute_iou( det, d->m_groundtruth[gi] );
        if( iou > best_iou && iou >= d->m_config.iou_threshold )
        {
          best_iou = iou;
          best_gt = gi;
          matched = true;
        }
      }

      if( matched )
      {
        gt_matched[best_gt] = true;
        tp++;
      }
      else
      {
        fp++;
      }
    }
    else
    {
      fp++;
    }

    if( det.confidence != prev_conf )
    {
      roc_curve_point rp;
      rp.confidence = det.confidence;
      rp.true_positive_rate = total_gt > 0 ?
        static_cast< double >( tp ) / total_gt : 0.0;
      // For object detection, FPR is often expressed as FP per image
      rp.false_positive_rate = total_frames > 0 ?
        static_cast< double >( fp ) / total_frames : 0.0;

      result.points.push_back( rp );
      prev_conf = det.confidence;
    }
  }

  // Add final point
  roc_curve_point final_pt;
  final_pt.confidence = 0.0;
  final_pt.true_positive_rate = total_gt > 0 ?
    static_cast< double >( tp ) / total_gt : 0.0;
  final_pt.false_positive_rate = total_frames > 0 ?
    static_cast< double >( fp ) / total_frames : 0.0;
  result.points.push_back( final_pt );

  // Compute AUC using trapezoidal rule
  double auc = 0.0;
  for( size_t i = 1; i < result.points.size(); i++ )
  {
    double dx = result.points[i].false_positive_rate -
                result.points[i-1].false_positive_rate;
    double avg_y = ( result.points[i].true_positive_rate +
                     result.points[i-1].true_positive_rate ) / 2.0;
    auc += dx * avg_y;
  }
  result.auc = auc;

  return result;
}

evaluation_plot_data
model_evaluator::generate_plot_data()
{
  evaluation_plot_data result;

  // Generate PR curves
  result.overall_pr_curve = generate_pr_curve();
  result.per_class_pr_curves = generate_per_class_pr_curves();

  // Generate confusion matrix
  result.confusion_matrix = generate_confusion_matrix();

  // Generate ROC curve
  result.overall_roc_curve = generate_roc_curve();

  // Generate histograms

  // IoU histogram (20 bins from 0 to 1)
  result.iou_histogram.resize( 20, 0 );
  for( const auto& frame_match_pair : d->m_frame_matches )
  {
    for( const auto& match : frame_match_pair.second.matches )
    {
      int bin = std::min( 19, static_cast< int >( match.iou * 20 ) );
      result.iou_histogram[bin]++;
    }
  }

  // Track purity histogram (10 bins from 0% to 100%)
  result.track_purity_histogram.resize( 10, 0 );

  // Track continuity histogram
  result.track_continuity_histogram.resize( 10, 0 );

  // Track length histogram
  for( const auto& pair : d->m_comp_track_lengths )
  {
    result.track_length_histogram[pair.second]++;
  }

  return result;
}

// =============================================================================
// Export functions
// =============================================================================

bool
model_evaluator::export_pr_curve_csv(
  const pr_curve_data& curve,
  const std::string& filepath )
{
  std::ofstream file( filepath );
  if( !file.is_open() )
  {
    return false;
  }

  file << "confidence,recall,precision,f1,tp,fp,fn\n";
  for( const auto& pt : curve.points )
  {
    file << std::fixed << std::setprecision( 6 )
         << pt.confidence << ","
         << pt.recall << ","
         << pt.precision << ","
         << pt.f1 << ","
         << pt.tp << ","
         << pt.fp << ","
         << pt.fn << "\n";
  }

  file.close();
  return true;
}

bool
model_evaluator::export_confusion_matrix_csv(
  const confusion_matrix_data& matrix,
  const std::string& filepath )
{
  std::ofstream file( filepath );
  if( !file.is_open() )
  {
    return false;
  }

  // Header row with class names
  file << "gt_class";
  for( const auto& name : matrix.class_names )
  {
    file << "," << name;
  }
  file << "\n";

  // Data rows
  for( size_t i = 0; i < matrix.class_names.size(); i++ )
  {
    file << matrix.class_names[i];
    for( size_t j = 0; j < matrix.class_names.size(); j++ )
    {
      file << "," << matrix.matrix[i][j];
    }
    file << "\n";
  }

  file.close();
  return true;
}

bool
model_evaluator::export_plot_data(
  const evaluation_plot_data& plot_data,
  const std::string& output_dir )
{
  namespace fs = std::filesystem;

  fs::path dir( output_dir );
  if( !fs::exists( dir ) )
  {
    std::error_code ec;
    if( !fs::create_directories( dir, ec ) )
    {
      return false;
    }
  }

  bool success = true;

  // Export overall PR curve
  success &= export_pr_curve_csv(
    plot_data.overall_pr_curve,
    ( dir / "pr_curve_overall.csv" ).string() );

  // Export per-class PR curves
  for( const auto& pair : plot_data.per_class_pr_curves )
  {
    std::string safe_name = pair.first;
    // Replace special characters in filename
    std::replace( safe_name.begin(), safe_name.end(), '/', '_' );
    std::replace( safe_name.begin(), safe_name.end(), '\\', '_' );
    std::replace( safe_name.begin(), safe_name.end(), ' ', '_' );

    success &= export_pr_curve_csv(
      pair.second,
      ( dir / ( "pr_curve_" + safe_name + ".csv" ) ).string() );
  }

  // Export confusion matrix
  success &= export_confusion_matrix_csv(
    plot_data.confusion_matrix,
    ( dir / "confusion_matrix.csv" ).string() );

  // Export ROC curve
  {
    std::ofstream file( ( dir / "roc_curve_overall.csv" ).string() );
    if( file.is_open() )
    {
      file << "confidence,fpr,tpr\n";
      for( const auto& pt : plot_data.overall_roc_curve.points )
      {
        file << std::fixed << std::setprecision( 6 )
             << pt.confidence << ","
             << pt.false_positive_rate << ","
             << pt.true_positive_rate << "\n";
      }
      file.close();
    }
    else
    {
      success = false;
    }
  }

  // Export histograms
  {
    std::ofstream file( ( dir / "histograms.csv" ).string() );
    if( file.is_open() )
    {
      // IoU histogram
      file << "iou_histogram\n";
      file << "bin_start,bin_end,count\n";
      for( size_t i = 0; i < plot_data.iou_histogram.size(); i++ )
      {
        file << std::fixed << std::setprecision( 2 )
             << ( i * 0.05 ) << "," << ( ( i + 1 ) * 0.05 ) << ","
             << plot_data.iou_histogram[i] << "\n";
      }

      // Track length histogram
      file << "\ntrack_length_histogram\n";
      file << "length,count\n";
      for( const auto& pair : plot_data.track_length_histogram )
      {
        file << pair.first << "," << pair.second << "\n";
      }

      file.close();
    }
    else
    {
      success = false;
    }
  }

  return success;
}

bool
model_evaluator::export_plot_data_json(
  const evaluation_plot_data& plot_data,
  const std::string& filepath )
{
  std::ofstream file( filepath );
  if( !file.is_open() )
  {
    return false;
  }

  file << "{\n";

  // Overall PR curve
  file << "  \"overall_pr_curve\": {\n";
  file << "    \"class_name\": \"" << plot_data.overall_pr_curve.class_name << "\",\n";
  file << "    \"average_precision\": " << plot_data.overall_pr_curve.average_precision << ",\n";
  file << "    \"max_f1\": " << plot_data.overall_pr_curve.max_f1 << ",\n";
  file << "    \"best_threshold\": " << plot_data.overall_pr_curve.best_threshold << ",\n";
  file << "    \"points\": [\n";
  for( size_t i = 0; i < plot_data.overall_pr_curve.points.size(); i++ )
  {
    const auto& pt = plot_data.overall_pr_curve.points[i];
    file << "      {\"recall\": " << pt.recall
         << ", \"precision\": " << pt.precision
         << ", \"confidence\": " << pt.confidence
         << ", \"f1\": " << pt.f1 << "}";
    if( i < plot_data.overall_pr_curve.points.size() - 1 )
      file << ",";
    file << "\n";
  }
  file << "    ]\n";
  file << "  },\n";

  // Per-class PR curves
  file << "  \"per_class_pr_curves\": {\n";
  size_t class_idx = 0;
  for( const auto& pair : plot_data.per_class_pr_curves )
  {
    file << "    \"" << pair.first << "\": {\n";
    file << "      \"average_precision\": " << pair.second.average_precision << ",\n";
    file << "      \"max_f1\": " << pair.second.max_f1 << ",\n";
    file << "      \"points\": [\n";
    for( size_t i = 0; i < pair.second.points.size(); i++ )
    {
      const auto& pt = pair.second.points[i];
      file << "        {\"recall\": " << pt.recall
           << ", \"precision\": " << pt.precision
           << ", \"confidence\": " << pt.confidence << "}";
      if( i < pair.second.points.size() - 1 )
        file << ",";
      file << "\n";
    }
    file << "      ]\n";
    file << "    }";
    if( ++class_idx < plot_data.per_class_pr_curves.size() )
      file << ",";
    file << "\n";
  }
  file << "  },\n";

  // Confusion matrix
  file << "  \"confusion_matrix\": {\n";
  file << "    \"class_names\": [";
  for( size_t i = 0; i < plot_data.confusion_matrix.class_names.size(); i++ )
  {
    file << "\"" << plot_data.confusion_matrix.class_names[i] << "\"";
    if( i < plot_data.confusion_matrix.class_names.size() - 1 )
      file << ", ";
  }
  file << "],\n";
  file << "    \"matrix\": [\n";
  for( size_t i = 0; i < plot_data.confusion_matrix.matrix.size(); i++ )
  {
    file << "      [";
    for( size_t j = 0; j < plot_data.confusion_matrix.matrix[i].size(); j++ )
    {
      file << plot_data.confusion_matrix.matrix[i][j];
      if( j < plot_data.confusion_matrix.matrix[i].size() - 1 )
        file << ", ";
    }
    file << "]";
    if( i < plot_data.confusion_matrix.matrix.size() - 1 )
      file << ",";
    file << "\n";
  }
  file << "    ],\n";
  file << "    \"overall_accuracy\": " << plot_data.confusion_matrix.overall_accuracy << "\n";
  file << "  },\n";

  // ROC curve
  file << "  \"overall_roc_curve\": {\n";
  file << "    \"auc\": " << plot_data.overall_roc_curve.auc << ",\n";
  file << "    \"points\": [\n";
  for( size_t i = 0; i < plot_data.overall_roc_curve.points.size(); i++ )
  {
    const auto& pt = plot_data.overall_roc_curve.points[i];
    file << "      {\"fpr\": " << pt.false_positive_rate
         << ", \"tpr\": " << pt.true_positive_rate
         << ", \"confidence\": " << pt.confidence << "}";
    if( i < plot_data.overall_roc_curve.points.size() - 1 )
      file << ",";
    file << "\n";
  }
  file << "    ]\n";
  file << "  },\n";

  // IoU histogram
  file << "  \"iou_histogram\": [";
  for( size_t i = 0; i < plot_data.iou_histogram.size(); i++ )
  {
    file << plot_data.iou_histogram[i];
    if( i < plot_data.iou_histogram.size() - 1 )
      file << ", ";
  }
  file << "],\n";

  // Track length histogram
  file << "  \"track_length_histogram\": {";
  size_t tl_idx = 0;
  for( const auto& pair : plot_data.track_length_histogram )
  {
    file << "\"" << pair.first << "\": " << pair.second;
    if( ++tl_idx < plot_data.track_length_histogram.size() )
      file << ", ";
  }
  file << "}\n";

  file << "}\n";

  file.close();
  return true;
}

} // namespace viame
