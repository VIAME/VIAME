/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Command-line tool for scoring detection/tracking results using evaluate_models

#include "score_results.h"

#include <evaluate_models.h>

#ifdef VIAME_TOOLS_HAVE_OPENCV
#include <plot_metrics.h>
#endif

#include <kwiversys/SystemTools.hxx>
#include <kwiversys/Directory.hxx>

#include <vital/plugin_management/plugin_manager.h>
#include <vital/logger/logger.h>

#include <vector>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <limits>
#include <memory>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

// =============================================================================
// Global variables and parameter class
// =============================================================================

class score_results_params
{
public:
  // General options
  bool opt_help = false;
  bool opt_verbose = false;

  // Input options
  std::string opt_computed;       // Computed detection/track files or folder
  std::string opt_truth;          // Ground truth files or folder
  std::string opt_input_ext;      // File extension filter

  // Scoring options
  double opt_iou_threshold = 0.5;
  double opt_confidence_threshold = 0.0;
  bool opt_per_class = false;
  bool opt_compute_tracking = true;

  // Output options
  std::string opt_output_metrics;    // Output metrics to JSON file
  std::string opt_output_summary;    // Output summary to text file
  std::string opt_output_plots;      // Output plot data to directory
  std::string opt_output_pr_csv;     // Output PR curve to CSV
  std::string opt_output_conf_csv;   // Output confusion matrix to CSV
  bool opt_json_curves = false;      // Inline curve points in the JSON
  std::string opt_labels;            // Class synonym file
  std::string opt_frame_list;        // Frame list to downselect scoring
  std::string opt_default_label;     // Label for detections carrying none
  bool opt_aux_confidence = false;   // Score on column 7 not the class score
  bool opt_top_class = false;        // Only each detection's best class
  bool opt_sweep = false;            // Sweep confidence thresholds
  int opt_sweep_interval = 100;      // Number of thresholds in the sweep
  std::string opt_filter_estimator = "min";  // DIVE filter estimate method
  std::string opt_output_sweep;      // Directory for sweep artifacts
  bool opt_track_detections = false; // Prefer *_tracks.csv over *_detections.csv
  std::string opt_input_format = "viame_csv";  // Reader for non-CSV inputs
  std::string opt_output_roc_csv;    // Output ROC curve to CSV
  bool opt_print_summary = true;

  score_results_params() = default;
  virtual ~score_results_params() = default;
};

static kwiver::vital::logger_handle_t g_logger;

// =============================================================================
// Helper functions
// =============================================================================

std::string escape_json( const std::string& input )
{
  std::string output;
  output.reserve( input.size() + 10 );

  for( char c : input )
  {
    switch( c )
    {
      case '"':  output += "\\\""; break;
      case '\\': output += "\\\\"; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default:   output += c; break;
    }
  }
  return output;
}

std::vector< std::string >
collect_files( const std::string& path, const std::string& ext )
{
  std::vector< std::string > files;

  if( kwiversys::SystemTools::FileIsDirectory( path ) )
  {
    kwiversys::Directory dir;
    if( dir.Load( path ) )
    {
      for( unsigned long i = 0; i < dir.GetNumberOfFiles(); ++i )
      {
        std::string filename = dir.GetFile( i );
        if( filename == "." || filename == ".." )
        {
          continue;
        }

        std::string file_ext = kwiversys::SystemTools::GetFilenameLastExtension( filename );
        if( ext.empty() || file_ext == ext )
        {
          files.push_back( path + "/" + filename );
        }
      }
    }
    std::sort( files.begin(), files.end() );
  }
  else if( kwiversys::SystemTools::FileExists( path ) )
  {
    files.push_back( path );
  }

  return files;
}

std::vector< std::pair< std::string, std::string > >
pair_files( const std::vector< std::string >& computed,
            const std::vector< std::string >& truth )
{
  std::vector< std::pair< std::string, std::string > > pairs;

  // Build map of truth files by base name
  std::map< std::string, std::string > truth_map;
  for( const auto& t : truth )
  {
    std::string base = kwiversys::SystemTools::GetFilenameWithoutLastExtension( t );
    truth_map[ base ] = t;
  }

  // Match computed files to truth files
  std::set< std::string > paired_truth;

  for( const auto& c : computed )
  {
    std::string base = kwiversys::SystemTools::GetFilenameWithoutLastExtension( c );

    auto it = truth_map.find( base );
    if( it != truth_map.end() )
    {
      pairs.push_back( { c, it->second } );
      paired_truth.insert( it->second );
    }
    else
    {
      LOG_WARN( g_logger, "No matching ground truth for: " << c );
    }
  }

  // Ground truth with no computed counterpart is excluded from scoring
  // entirely, which removes its objects from the false negative count and so
  // makes recall look better than it is. Say so rather than dropping silently.
  for( const auto& t : truth )
  {
    if( !paired_truth.count( t ) )
    {
      LOG_WARN( g_logger, "No computed results for ground truth: " << t
                << " (its annotations are excluded from all metrics)" );
    }
  }

  return pairs;
}

void print_summary( const viame::evaluation_results& results )
{
  std::cout << "\n";
  std::cout << "===============================================================================\n";
  std::cout << "                           EVALUATION SUMMARY\n";
  std::cout << "===============================================================================\n\n";

  std::cout << "--- Dataset Statistics ---\n";
  std::cout << "  Total frames:           " << static_cast< int >( results.total_frames ) << "\n";
  std::cout << "  Ground truth objects:   " << static_cast< int >( results.total_gt_objects ) << "\n";
  std::cout << "  Computed detections:    " << static_cast< int >( results.total_computed ) << "\n";
  std::cout << "  Ground truth tracks:    " << static_cast< int >( results.total_gt_tracks ) << "\n";
  std::cout << "  Computed tracks:        " << static_cast< int >( results.total_computed_tracks ) << "\n";
  std::cout << "\n";

  std::cout << "--- Detection Metrics ---\n";
  std::cout << std::fixed << std::setprecision( 4 );
  std::cout << "  True Positives:         " << static_cast< int >( results.true_positives ) << "\n";
  std::cout << "  False Positives:        " << static_cast< int >( results.false_positives ) << "\n";
  std::cout << "  False Negatives:        " << static_cast< int >( results.false_negatives ) << "\n";
  std::cout << "  Precision:              " << results.precision << "\n";
  std::cout << "  Recall:                 " << results.recall << "\n";
  std::cout << "  F1 Score:               " << results.f1_score << "\n";
  std::cout << "  MCC:                    " << results.mcc << "\n";
  std::cout << "  Average Precision:      " << results.average_precision << "\n";
  std::cout << "  AP@any:                 " << results.ap_any << "\n";
  std::cout << "  AP@50:                  " << results.ap50 << "\n";
  std::cout << "  AP@75:                  " << results.ap75 << "\n";
  std::cout << "  AP@50:95:               " << results.ap50_95 << "\n";
  std::cout << "\n";

  std::cout << "--- Localization Quality ---\n";
  std::cout << "  Mean IoU:               " << results.mean_iou << "\n";
  std::cout << "  Median IoU:             " << results.median_iou << "\n";
  std::cout << "  Mean Center Distance:   " << results.mean_center_distance << " px\n";
  std::cout << "  Mean Size Error:        " << results.mean_size_error << "\n";
  std::cout << "\n";

  if( results.total_gt_tracks > 0 || results.total_computed_tracks > 0 )
  {
    std::cout << "--- MOT Tracking Metrics ---\n";
    std::cout << "  MOTA:                   " << results.mota << "\n";
    std::cout << "  MOTP:                   " << results.motp << "\n";
    std::cout << "  IDF1:                   " << results.idf1 << "\n";
    std::cout << "  ID Precision:           " << results.idp << "\n";
    std::cout << "  ID Recall:              " << results.idr << "\n";
    std::cout << "  ID Switches:            " << static_cast< int >( results.id_switches ) << "\n";
    std::cout << "  Fragmentations:         " << static_cast< int >( results.fragmentations ) << "\n";
    std::cout << "  Mostly Tracked:         " << static_cast< int >( results.mostly_tracked )
              << " (" << results.mt_ratio * 100 << "%)\n";
    std::cout << "  Partially Tracked:      " << static_cast< int >( results.partially_tracked )
              << " (" << results.pt_ratio * 100 << "%)\n";
    std::cout << "  Mostly Lost:            " << static_cast< int >( results.mostly_lost )
              << " (" << results.ml_ratio * 100 << "%)\n";
    std::cout << "  False Alarms/Frame:     " << results.faf << "\n";
    std::cout << "\n";

    std::cout << "--- HOTA Metrics ---\n";
    std::cout << "  HOTA:                   " << results.hota << "\n";
    std::cout << "  DetA:                   " << results.deta << "\n";
    std::cout << "  AssA:                   " << results.assa << "\n";
    std::cout << "  LocA:                   " << results.loca << "\n";
    std::cout << "\n";

    std::cout << "--- KWANT-style Metrics ---\n";
    std::cout << "  Track Continuity:       " << results.avg_track_continuity << "\n";
    std::cout << "  Track Purity:           " << results.avg_track_purity << "\n";
    std::cout << "  Target Continuity:      " << results.avg_target_continuity << "\n";
    std::cout << "  Target Purity:          " << results.avg_target_purity << "\n";
    std::cout << "  Track Pd:               " << results.track_pd << "\n";
    std::cout << "  Track FA:               " << results.track_fa << "\n";
    std::cout << "\n";

    std::cout << "--- Track Quality ---\n";
    std::cout << "  Avg Track Length:       " << results.avg_track_length << " frames\n";
    std::cout << "  Avg GT Track Length:    " << results.avg_gt_track_length << " frames\n";
    std::cout << "  Track Completeness:     " << results.track_completeness << "\n";
    std::cout << "  Avg Gap Length:         " << results.avg_gap_length << " frames\n";
    std::cout << "\n";
  }

  std::cout << "--- Classification ---\n";
  std::cout << "  Classification Acc:     " << results.classification_accuracy << "\n";
  std::cout << "  Mean AP (per-class):    " << results.mean_ap << "\n";
  std::cout << "\n";

  std::cout << "===============================================================================\n\n";
}

void print_per_class_metrics( const viame::evaluation_results& results )
{
  if( results.per_class_metrics.empty() )
  {
    return;
  }

  std::cout << "--- Per-Class Metrics ---\n\n";

  // Header
  std::cout << std::left << std::setw( 25 ) << "Class"
            << std::right << std::setw( 10 ) << "TP"
            << std::setw( 10 ) << "FP"
            << std::setw( 10 ) << "FN"
            << std::setw( 12 ) << "Precision"
            << std::setw( 12 ) << "Recall"
            << std::setw( 12 ) << "F1"
            << std::setw( 12 ) << "AP@any"
            << std::setw( 12 ) << "AP50"
            << std::setw( 12 ) << "AP75"
            << std::setw( 12 ) << "AP[.5:.95]"
            << "\n";
  std::cout << std::string( 139, '-' ) << "\n";

  for( const auto& kv : results.per_class_metrics )
  {
    const auto& class_name = kv.first;
    const auto& metrics = kv.second;

    auto get_metric = [&]( const std::string& name ) -> double
    {
      auto it = metrics.find( name );
      return ( it != metrics.end() ) ? it->second : 0.0;
    };

    // Print a metric the evaluator did not produce as "n/a": defaulting it to
    // zero would be indistinguishable from a genuine score of zero
    auto format_metric = [&]( const std::string& name ) -> std::string
    {
      auto it = metrics.find( name );
      if( it == metrics.end() )
      {
        return "n/a";
      }

      std::ostringstream oss;
      oss << std::fixed << std::setprecision( 4 ) << it->second;
      return oss.str();
    };

    std::cout << std::left << std::setw( 25 ) << class_name
              << std::right << std::setw( 10 ) << static_cast< int >( get_metric( "true_positives" ) )
              << std::setw( 10 ) << static_cast< int >( get_metric( "false_positives" ) )
              << std::setw( 10 ) << static_cast< int >( get_metric( "false_negatives" ) )
              << std::setw( 12 ) << format_metric( "precision" )
              << std::setw( 12 ) << format_metric( "recall" )
              << std::setw( 12 ) << format_metric( "f1_score" )
              << std::setw( 12 ) << format_metric( "ap_any" )
              << std::setw( 12 ) << format_metric( "ap50" )
              << std::setw( 12 ) << format_metric( "ap75" )
              << std::setw( 12 ) << format_metric( "ap50_95" )
              << "\n";
  }
  std::cout << "\n";
}

// The confusion matrix is O(classes^2) and always small, so it goes into the
// metrics JSON unconditionally. The curves are one point per detection with no
// downsampling -- a large run produces millions -- so they are opt-in behind
// --json-curves rather than silently turning a metrics file into a huge one.
// Synonym file: "canonical: alias1, alias2" per line, blanks and # ignored.
// Aliases map onto the canonical name; the canonical name maps to itself so a
// file may list it explicitly without surprise.
bool load_label_synonyms( const std::string& path,
                          std::map< std::string, std::string >& out )
{
  std::ifstream in( path );
  if( !in.is_open() )
  {
    LOG_ERROR( g_logger, "Could not open labels file: " << path );
    return false;
  }

  auto trim = []( std::string v ) -> std::string
  {
    const size_t b = v.find_first_not_of( " \t\r\n" );
    const size_t e = v.find_last_not_of( " \t\r\n" );
    return ( b == std::string::npos ) ? std::string() : v.substr( b, e - b + 1 );
  };

  std::string line;
  while( std::getline( in, line ) )
  {
    line = trim( line );
    if( line.empty() || line[0] == '#' )
    {
      continue;
    }

    const size_t colon = line.find( ':' );
    const std::string canonical =
      trim( colon == std::string::npos ? line : line.substr( 0, colon ) );
    if( canonical.empty() )
    {
      continue;
    }
    out[canonical] = canonical;

    if( colon == std::string::npos )
    {
      continue;
    }

    std::stringstream aliases( line.substr( colon + 1 ) );
    std::string alias;
    while( std::getline( aliases, alias, ',' ) )
    {
      alias = trim( alias );
      if( !alias.empty() )
      {
        out[alias] = canonical;
      }
    }
  }

  LOG_INFO( g_logger, "Loaded " << out.size() << " label mappings from " << path );
  return true;
}

bool load_frame_list( const std::string& path, std::set< std::string >& out )
{
  std::ifstream in( path );
  if( !in.is_open() )
  {
    LOG_ERROR( g_logger, "Could not open frame list: " << path );
    return false;
  }

  std::string line;
  while( std::getline( in, line ) )
  {
    const size_t b = line.find_first_not_of( " \t\r\n" );
    const size_t e = line.find_last_not_of( " \t\r\n" );
    if( b == std::string::npos )
    {
      continue;
    }
    line = line.substr( b, e - b + 1 );
    if( !line.empty() && line[0] != '#' )
    {
      out.insert( line );
    }
  }

  LOG_INFO( g_logger, "Scoring restricted to " << out.size()
            << " frames from " << path );
  return true;
}

// A VIAME output folder commonly holds both <seq>_detections.csv and
// <seq>_tracks.csv for the same sequence. Scoring both would double count, so
// one form is chosen per sequence: detections by default, tracks when asked.
std::vector< std::string >
select_track_or_detection_files( const std::vector< std::string >& files,
                                 bool prefer_tracks )
{
  auto stem_of = []( const std::string& path ) -> std::string
  {
    std::string base =
      kwiversys::SystemTools::GetFilenameWithoutLastExtension( path );
    for( const char* suffix : { "_detections", "_tracks" } )
    {
      const size_t n = std::string( suffix ).size();
      if( base.size() > n && base.compare( base.size() - n, n, suffix ) == 0 )
      {
        return base.substr( 0, base.size() - n );
      }
    }
    return base;
  };

  auto is_tracks = []( const std::string& path ) -> bool
  {
    const std::string base =
      kwiversys::SystemTools::GetFilenameWithoutLastExtension( path );
    return base.size() > 7 &&
           base.compare( base.size() - 7, 7, "_tracks" ) == 0;
  };

  std::map< std::string, std::string > chosen;
  std::vector< std::string > passthrough;

  for( const auto& f : files )
  {
    const std::string base =
      kwiversys::SystemTools::GetFilenameWithoutLastExtension( f );
    const bool tracks = is_tracks( f );
    const bool dets = base.size() > 11 &&
                      base.compare( base.size() - 11, 11, "_detections" ) == 0;

    if( !tracks && !dets )
    {
      passthrough.push_back( f );
      continue;
    }

    const std::string stem = stem_of( f );
    auto it = chosen.find( stem );
    if( it == chosen.end() )
    {
      chosen[stem] = f;
    }
    else if( tracks == prefer_tracks )
    {
      it->second = f;  // the preferred form wins the slot
    }
  }

  std::vector< std::string > out = passthrough;
  for( const auto& kv : chosen )
  {
    out.push_back( kv.second );
  }
  std::sort( out.begin(), out.end() );
  return out;
}

// Per class, the threshold maximising IDF1 and the one maximising MOTA.
// Ordered [idf1, idf1_thresh, mota, mota_thresh] to match the columns the DIVE
// side has always consumed.
struct sweep_result
{
  double idf1 = -1.0;
  double idf1_thresh = 0.0;
  double mota = -std::numeric_limits< double >::max();
  double mota_thresh = 0.0;
};

// Turn swept thresholds into the per-class confidence filter DIVE applies.
// "min" is deliberately the conservative choice: it keeps whichever of the two
// operating points admits more detections, so the filter never hides anything
// either metric wanted.
bool write_dive_filter( const std::string& filepath,
                        const std::map< std::string, sweep_result >& scores,
                        const std::string& method )
{
  std::map< std::string, double > filters;

  for( const auto& kv : scores )
  {
    const auto& v = kv.second;
    double value = 0.0;

    if( method == "min" )
    {
      value = std::min( v.idf1_thresh, v.mota_thresh );
    }
    else if( method == "avg" || method == "avg_minus_1p" )
    {
      const double adj = ( method == "avg_minus_1p" ) ? -0.01 : 0.0;
      value = std::max( 0.5 * ( v.idf1_thresh + v.mota_thresh ) + adj, 0.0 );
    }
    else if( method == "idf1" )
    {
      value = v.idf1_thresh;
    }
    else if( method == "mota" )
    {
      value = v.mota_thresh;
    }
    else
    {
      LOG_ERROR( g_logger, "Unknown filter estimator: " << method );
      return false;
    }

    filters[kv.first] = value;
  }

  // DIVE falls back to "default" for any class the file does not name, so
  // supply the least aggressive per-class filter rather than leaving unlisted
  // classes unfiltered.
  if( !filters.count( "default" ) && !filters.empty() )
  {
    double min_filter = std::numeric_limits< double >::max();
    for( const auto& kv : filters )
    {
      min_filter = std::min( min_filter, kv.second );
    }
    if( min_filter > 0.0 )
    {
      filters["default"] = min_filter;
    }
  }

  std::ofstream out( filepath );
  if( !out.is_open() )
  {
    LOG_ERROR( g_logger, "Could not open filter file: " << filepath );
    return false;
  }

  out << std::fixed << std::setprecision( 6 );
  out << "{\n    \"confidenceFilters\": {\n";
  bool first = true;
  for( const auto& kv : filters )
  {
    if( !first ) out << ",\n";
    out << "        \"" << escape_json( kv.first ) << "\": " << kv.second;
    first = false;
  }
  out << "\n    }\n}\n";
  out.close();

  LOG_INFO( g_logger, "DIVE confidence filter written to: " << filepath );
  return true;
}

bool write_metrics_json( const viame::evaluation_results& results,
                         const std::string& filepath,
                         const viame::evaluation_plot_data* plot_data = nullptr,
                         bool include_curves = false )
{
  std::ofstream out( filepath );
  if( !out.is_open() )
  {
    LOG_ERROR( g_logger, "Could not open output file: " << filepath );
    return false;
  }

  out << std::fixed << std::setprecision( 6 );
  out << "{\n";

  // A non-finite value would be streamed as a bare nan or inf token, which is
  // not valid JSON; emit null instead
  auto json_value = []( double value ) -> std::string
  {
    if( !std::isfinite( value ) )
    {
      return "null";
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision( 6 ) << value;
    return oss.str();
  };

  bool first = true;
  for( const auto& kv : results.all_metrics )
  {
    if( !first ) out << ",\n";
    out << "  \"" << escape_json( kv.first ) << "\": " << json_value( kv.second );
    first = false;
  }

  // Add per-class metrics if present
  if( !results.per_class_metrics.empty() )
  {
    out << ",\n  \"per_class\": {\n";

    bool first_class = true;
    for( const auto& class_kv : results.per_class_metrics )
    {
      if( !first_class ) out << ",\n";
      out << "    \"" << escape_json( class_kv.first ) << "\": {";

      bool first_metric = true;
      for( const auto& metric_kv : class_kv.second )
      {
        if( !first_metric ) out << ", ";
        out << "\"" << escape_json( metric_kv.first ) << "\": "
            << json_value( metric_kv.second );
        first_metric = false;
      }

      out << "}";
      first_class = false;
    }

    out << "\n  }";
  }

  if( plot_data )
  {
    const auto& cm = plot_data->confusion_matrix;

    if( !cm.class_names.empty() )
    {
      out << ",\n  \"confusion_matrix\": {\n";
      out << "    \"class_names\": [";
      for( size_t i = 0; i < cm.class_names.size(); ++i )
      {
        if( i ) out << ", ";
        out << "\"" << escape_json( cm.class_names[i] ) << "\"";
      }
      out << "],\n";

      out << "    \"matrix\": [";
      for( size_t r = 0; r < cm.matrix.size(); ++r )
      {
        if( r ) out << ", ";
        out << "[";
        for( size_t c = 0; c < cm.matrix[r].size(); ++c )
        {
          if( c ) out << ", ";
          out << cm.matrix[r][c];
        }
        out << "]";
      }
      out << "],\n";

      out << "    \"normalized_matrix\": [";
      for( size_t r = 0; r < cm.normalized_matrix.size(); ++r )
      {
        if( r ) out << ", ";
        out << "[";
        for( size_t c = 0; c < cm.normalized_matrix[r].size(); ++c )
        {
          if( c ) out << ", ";
          out << json_value( cm.normalized_matrix[r][c] );
        }
        out << "]";
      }
      out << "],\n";

      out << "    \"per_class_accuracy\": {";
      bool first_acc = true;
      for( const auto& kv : cm.per_class_accuracy )
      {
        if( !first_acc ) out << ", ";
        out << "\"" << escape_json( kv.first ) << "\": " << json_value( kv.second );
        first_acc = false;
      }
      out << "}\n  }";
    }

    if( include_curves )
    {
      auto write_pr = [&]( const viame::pr_curve_data& pr, const char* indent )
      {
        out << "{\n";
        out << indent << "  \"average_precision\": "
            << json_value( pr.average_precision ) << ",\n";
        out << indent << "  \"max_f1\": " << json_value( pr.max_f1 ) << ",\n";
        out << indent << "  \"best_threshold\": "
            << json_value( pr.best_threshold ) << ",\n";
        out << indent << "  \"points\": [\n";
        for( size_t i = 0; i < pr.points.size(); ++i )
        {
          const auto& pt = pr.points[i];
          out << indent << "    {\"recall\": " << json_value( pt.recall )
              << ", \"precision\": " << json_value( pt.precision )
              << ", \"confidence\": " << json_value( pt.confidence )
              << ", \"f1\": " << json_value( pt.f1 )
              << ", \"tp\": " << pt.tp
              << ", \"fp\": " << pt.fp
              << ", \"fn\": " << pt.fn << "}";
          if( i + 1 < pr.points.size() ) out << ",";
          out << "\n";
        }
        out << indent << "  ]\n" << indent << "}";
      };

      out << ",\n  \"pr_curve\": ";
      write_pr( plot_data->overall_pr_curve, "  " );

      out << ",\n  \"roc_curve\": {\n";
      out << "    \"mean_pd\": "
          << json_value( plot_data->overall_roc_curve.mean_pd ) << ",\n";
      out << "    \"max_false_alarms_per_frame\": "
          << json_value( plot_data->overall_roc_curve.max_false_alarms_per_frame )
          << ",\n";
      out << "    \"points\": [\n";
      for( size_t i = 0; i < plot_data->overall_roc_curve.points.size(); ++i )
      {
        const auto& pt = plot_data->overall_roc_curve.points[i];
        out << "      {\"false_alarms_per_frame\": "
            << json_value( pt.false_alarms_per_frame )
            << ", \"true_positive_rate\": "
            << json_value( pt.true_positive_rate )
            << ", \"confidence\": " << json_value( pt.confidence ) << "}";
        if( i + 1 < plot_data->overall_roc_curve.points.size() ) out << ",";
        out << "\n";
      }
      out << "    ]\n  }";

      if( !plot_data->per_class_pr_curves.empty() )
      {
        out << ",\n  \"per_class_pr_curves\": {\n";
        bool first_curve = true;
        for( const auto& kv : plot_data->per_class_pr_curves )
        {
          if( !first_curve ) out << ",\n";
          out << "    \"" << escape_json( kv.first ) << "\": ";
          write_pr( kv.second, "    " );
          first_curve = false;
        }
        out << "\n  }";
      }
    }
  }

  out << "\n}\n";
  out.close();

  LOG_INFO( g_logger, "Metrics written to: " << filepath );
  return true;
}

bool write_summary_text( const viame::evaluation_results& results,
                         const std::string& filepath )
{
  std::ofstream out( filepath );
  if( !out.is_open() )
  {
    LOG_ERROR( g_logger, "Could not open output file: " << filepath );
    return false;
  }

  // Redirect stdout to the file temporarily
  std::streambuf* old_buf = std::cout.rdbuf( out.rdbuf() );
  print_summary( results );
  if( !results.per_class_metrics.empty() )
  {
    print_per_class_metrics( results );
  }
  std::cout.rdbuf( old_buf );

  out.close();

  LOG_INFO( g_logger, "Summary written to: " << filepath );
  return true;
}

// =============================================================================
// Main entry point
// =============================================================================

namespace viame {
namespace tools {

// =============================================================================
void
score_results_applet
::add_command_options()
{
  m_cmd_options->add_options()
    ( "h,help", "Display usage information",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "v,verbose", "Enable verbose output",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "c,computed", "Input computed detection/track file or folder",
      ::cxxopts::value< std::string >()->default_value( "" ), "path" )
    ( "t,truth", "Input ground truth file or folder",
      ::cxxopts::value< std::string >()->default_value( "" ), "path" )
    ( "input-ext", "File extension filter for folder inputs (default: .csv)",
      ::cxxopts::value< std::string >()->default_value( "" ), "ext" )
    ( "iou-threshold", "IoU threshold for matching",
      ::cxxopts::value< double >()->default_value( "0.5" ), "value" )
    ( "iou", "IoU threshold for matching",
      ::cxxopts::value< double >()->default_value( "0.5" ), "value" )
    ( "confidence-threshold", "Minimum confidence threshold",
      ::cxxopts::value< double >()->default_value( "0.0" ), "value" )
    ( "conf", "Minimum confidence threshold",
      ::cxxopts::value< double >()->default_value( "0.0" ), "value" )
    ( "per-class", "Compute per-class metrics",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "no-tracking", "Disable tracking metrics computation",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "o,output-metrics", "Output all metrics to JSON file",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "output-summary", "Output summary to text file",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "output-plots", "Output plot data (PR, ROC, confusion) to directory",
      ::cxxopts::value< std::string >()->default_value( "" ), "dir" )
    ( "output-pr-csv", "Output precision-recall curve to CSV",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "output-conf-csv", "Output confusion matrix to CSV",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "output-roc-csv", "Output ROC curve to CSV",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "input-format", "Input file format: viame_csv (default) or any kwiver "
      "reader such as coco, cvat, dive, habcam, yolo",
      ::cxxopts::value< std::string >()->default_value( "viame_csv" ), "name" )
    ( "track-detections", "In a VIAME folder holding both *_detections.csv "
      "and *_tracks.csv, score the detections stored in the track files "
      "instead of the detection files",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "sweep-thresholds", "Score at a range of confidence thresholds and "
      "report, per class, the threshold maximising IDF1 and the one "
      "maximising MOTA",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "sweep-interval", "Number of thresholds in the sweep (default: 100, "
      "i.e. 0.00 to 0.99)",
      ::cxxopts::value< int >()->default_value( "100" ), "count" )
    ( "filter-estimator", "How to turn the swept thresholds into a DIVE "
      "confidence filter: none, min, avg, avg_minus_1p, idf1, mota",
      ::cxxopts::value< std::string >()->default_value( "min" ), "name" )
    ( "output-sweep", "Directory for sweep output: class_metrics.csv and, "
      "unless the estimator is none, dive.config.json",
      ::cxxopts::value< std::string >()->default_value( "" ), "dir" )
    ( "labels", "Class synonym file mapping alternate names onto canonical "
      "ones, so a model and its groundtruth may use different vocabularies. "
      "One class per line: 'canonical: alias1, alias2'",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "list", "Text file of frame identifiers, one per line. Only these "
      "frames are scored, on both sides",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "defaultlabel", "Class name to report for detections that carry none",
      ::cxxopts::value< std::string >()->default_value( "" ), "name" )
    ( "aux-confidence", "Rank and threshold on the detection confidence "
      "column rather than the per-class score",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "top-class", "In per-class scoring consider only each detection's "
      "highest scoring class, instead of every class it names",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "json-curves", "Include full PR and ROC curve points in the metrics "
      "JSON. Off by default: curves carry one point per detection, so a "
      "large run inlines millions",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "no-print", "Suppress printing summary to stdout",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ;
}

// =============================================================================
int
score_results_applet
::run()
{
  g_logger = kwiver::vital::get_logger( "viame.tools.score_results" );

  auto& cmd_args = command_args();

  if( cmd_args[ "help" ].as< bool >() )
  {
    std::cout << "Usage: viame score [options]\n\n"
              << "Score detection and tracking results using the evaluate_models library.\n"
              << "Computes comprehensive metrics including precision, recall, F1, AP,\n"
              << "MOT metrics (MOTA, MOTP, IDF1), HOTA, and KWANT-style metrics.\n"
              << m_cmd_options->help()
              << "\nExamples:\n"
              << "  viame score -c detections.csv -t groundtruth.csv\n"
              << "  viame score -c results/ -t truth/ --iou 0.5 --per-class\n"
              << "  viame score -c det.csv -t gt.csv -o metrics.json --output-plots plots/\n"
              << "  viame score -c det.csv -t gt.csv --output-pr-csv pr_curve.csv\n"
              << std::endl;
    return EXIT_SUCCESS;
  }

  score_results_params params;

  params.opt_verbose = cmd_args[ "verbose" ].as< bool >();
  params.opt_computed = cmd_args[ "computed" ].as< std::string >();
  params.opt_truth = cmd_args[ "truth" ].as< std::string >();
  params.opt_input_ext = cmd_args[ "input-ext" ].as< std::string >();

  // --iou and --iou-threshold are aliases, as are --conf and
  // --confidence-threshold; take whichever was actually given
  params.opt_iou_threshold = cmd_args.count( "iou" )
    ? cmd_args[ "iou" ].as< double >()
    : cmd_args[ "iou-threshold" ].as< double >();
  params.opt_confidence_threshold = cmd_args.count( "conf" )
    ? cmd_args[ "conf" ].as< double >()
    : cmd_args[ "confidence-threshold" ].as< double >();

  params.opt_per_class = cmd_args[ "per-class" ].as< bool >();
  params.opt_compute_tracking = !cmd_args[ "no-tracking" ].as< bool >();
  params.opt_output_metrics = cmd_args[ "output-metrics" ].as< std::string >();
  params.opt_output_summary = cmd_args[ "output-summary" ].as< std::string >();
  params.opt_output_plots = cmd_args[ "output-plots" ].as< std::string >();
  params.opt_output_pr_csv = cmd_args[ "output-pr-csv" ].as< std::string >();
  params.opt_output_conf_csv = cmd_args[ "output-conf-csv" ].as< std::string >();
  params.opt_output_roc_csv = cmd_args[ "output-roc-csv" ].as< std::string >();
  params.opt_input_format = cmd_args[ "input-format" ].as< std::string >();
  params.opt_track_detections = cmd_args[ "track-detections" ].as< bool >();
  params.opt_sweep = cmd_args[ "sweep-thresholds" ].as< bool >();
  params.opt_sweep_interval = cmd_args[ "sweep-interval" ].as< int >();
  params.opt_filter_estimator = cmd_args[ "filter-estimator" ].as< std::string >();
  params.opt_output_sweep = cmd_args[ "output-sweep" ].as< std::string >();
  params.opt_labels = cmd_args[ "labels" ].as< std::string >();
  params.opt_frame_list = cmd_args[ "list" ].as< std::string >();
  params.opt_default_label = cmd_args[ "defaultlabel" ].as< std::string >();
  params.opt_aux_confidence = cmd_args[ "aux-confidence" ].as< bool >();
  params.opt_top_class = cmd_args[ "top-class" ].as< bool >();
  params.opt_json_curves = cmd_args[ "json-curves" ].as< bool >();
  params.opt_print_summary = !cmd_args[ "no-print" ].as< bool >();

  // Validate inputs
  if( params.opt_computed.empty() )
  {
    LOG_ERROR( g_logger, "No computed file/folder specified. Use --computed or -c option." );
    return EXIT_FAILURE;
  }

  if( params.opt_truth.empty() )
  {
    LOG_ERROR( g_logger, "No ground truth file/folder specified. Use --truth or -t option." );
    return EXIT_FAILURE;
  }

  if( !kwiversys::SystemTools::FileExists( params.opt_computed ) )
  {
    LOG_ERROR( g_logger, "Computed path does not exist: " << params.opt_computed );
    return EXIT_FAILURE;
  }

  if( !kwiversys::SystemTools::FileExists( params.opt_truth ) )
  {
    LOG_ERROR( g_logger, "Ground truth path does not exist: " << params.opt_truth );
    return EXIT_FAILURE;
  }

  // Set default extension
  if( params.opt_input_ext.empty() )
  {
    params.opt_input_ext = ".csv";
  }
  else if( params.opt_input_ext[0] != '.' )
  {
    params.opt_input_ext = "." + params.opt_input_ext;
  }

  // Load plugins (needed for CSV readers)
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  // Collect input files
  auto computed_files = select_track_or_detection_files(
    collect_files( params.opt_computed, params.opt_input_ext ),
    params.opt_track_detections );
  auto truth_files = collect_files( params.opt_truth, params.opt_input_ext );

  if( computed_files.empty() )
  {
    LOG_ERROR( g_logger, "No computed files found in: " << params.opt_computed );
    return EXIT_FAILURE;
  }

  if( truth_files.empty() )
  {
    LOG_ERROR( g_logger, "No ground truth files found in: " << params.opt_truth );
    return EXIT_FAILURE;
  }

  // Pair files by basename if directories were provided
  std::vector< std::string > final_computed;
  std::vector< std::string > final_truth;

  if( computed_files.size() == 1 && truth_files.size() == 1 )
  {
    // Single file mode
    final_computed = computed_files;
    final_truth = truth_files;
  }
  else
  {
    // Directory mode - pair by basename
    auto pairs = pair_files( computed_files, truth_files );

    if( pairs.empty() )
    {
      LOG_ERROR( g_logger, "No matching file pairs found between computed and truth directories" );
      return EXIT_FAILURE;
    }

    for( const auto& p : pairs )
    {
      final_computed.push_back( p.first );
      final_truth.push_back( p.second );
    }
  }

  LOG_INFO( g_logger, "Evaluating " << final_computed.size() << " file pair(s)..." );

  if( params.opt_verbose )
  {
    for( size_t i = 0; i < final_computed.size(); ++i )
    {
      LOG_INFO( g_logger, "  " << final_computed[i] << " <-> " << final_truth[i] );
    }
  }

  // Configure evaluation
  viame::evaluation_config config;
  config.iou_threshold = params.opt_iou_threshold;
  config.confidence_threshold = params.opt_confidence_threshold;
  config.compute_tracking_metrics = params.opt_compute_tracking;
  config.compute_per_class_metrics = params.opt_per_class;
  config.use_aux_confidence = params.opt_aux_confidence;
  config.top_class_only = params.opt_top_class;
  config.default_label = params.opt_default_label;
  config.input_format = params.opt_input_format;

  if( !params.opt_labels.empty() &&
      !load_label_synonyms( params.opt_labels, config.label_synonyms ) )
  {
    return EXIT_FAILURE;
  }

  if( !params.opt_frame_list.empty() &&
      !load_frame_list( params.opt_frame_list, config.frame_whitelist ) )
  {
    return EXIT_FAILURE;
  }

  // Create evaluator and run evaluation
  viame::model_evaluator evaluator;
  evaluator.set_config( config );

  viame::evaluation_results results;
  try
  {
    results = evaluator.evaluate( final_computed, final_truth );
  }
  catch( const std::exception& e )
  {
    LOG_ERROR( g_logger, "Evaluation failed: " << e.what() );
    return EXIT_FAILURE;
  }

  // Print summary to stdout
  if( params.opt_print_summary )
  {
    print_summary( results );
    if( params.opt_per_class )
    {
      print_per_class_metrics( results );
    }
  }

  // Write outputs
  bool success = true;

  // Threshold sweep. evaluate() has already loaded the inputs, so each step
  // re-filters that copy rather than re-parsing; cost is thresholds x classes
  // evaluations, which --sweep-interval controls.
  if( params.opt_sweep )
  {
    if( params.opt_sweep_interval < 1 )
    {
      LOG_ERROR( g_logger, "--sweep-interval must be at least 1" );
      return EXIT_FAILURE;
    }

    std::set< std::string > sweep_classes;
    for( const auto& kv : results.per_class_metrics )
    {
      sweep_classes.insert( kv.first );
    }
    if( sweep_classes.empty() )
    {
      // Without --per-class there is nothing to break down, so sweep the
      // aggregate and report it under a single name.
      sweep_classes.insert( std::string() );
    }

    std::map< std::string, sweep_result > sweep_scores;

    LOG_INFO( g_logger, "Sweeping " << params.opt_sweep_interval
              << " thresholds over " << sweep_classes.size() << " class(es)..." );

    for( const auto& class_name : sweep_classes )
    {
      sweep_result best;

      for( int i = 0; i < params.opt_sweep_interval; ++i )
      {
        const double thresh =
          static_cast< double >( i ) / params.opt_sweep_interval;

        const auto r = evaluator.evaluate_loaded( thresh, class_name );

        if( r.idf1 > best.idf1 )
        {
          best.idf1 = r.idf1;
          best.idf1_thresh = thresh;
        }
        if( r.mota > best.mota )
        {
          best.mota = r.mota;
          best.mota_thresh = thresh;
        }
      }

      sweep_scores[class_name.empty() ? "default" : class_name] = best;
    }

    const std::string sweep_dir = params.opt_output_sweep.empty()
      ? std::string( "." ) : params.opt_output_sweep;

    if( !params.opt_output_sweep.empty() &&
        !kwiversys::SystemTools::FileIsDirectory( sweep_dir ) &&
        !kwiversys::SystemTools::MakeDirectory( sweep_dir ) )
    {
      LOG_ERROR( g_logger, "Could not create sweep output directory: " << sweep_dir );
      return EXIT_FAILURE;
    }

    const std::string csv_path = sweep_dir + "/class_metrics.csv";
    std::ofstream csv( csv_path );
    if( csv.is_open() )
    {
      csv << std::fixed << std::setprecision( 6 );
      csv << "# class,idf1,idf1_thresh,mota,mota_thresh\n";
      for( const auto& kv : sweep_scores )
      {
        csv << kv.first << "," << kv.second.idf1 << "," << kv.second.idf1_thresh
            << "," << kv.second.mota << "," << kv.second.mota_thresh << "\n";
      }
      csv.close();
      LOG_INFO( g_logger, "Sweep metrics written to: " << csv_path );
    }
    else
    {
      LOG_ERROR( g_logger, "Could not open sweep output: " << csv_path );
      success = false;
    }

    std::cout << "\n--- Threshold Sweep ---\n";
    std::cout << std::fixed << std::setprecision( 4 );
    for( const auto& kv : sweep_scores )
    {
      std::cout << "  " << kv.first
                << ": IDF1 " << kv.second.idf1
                << " @ " << kv.second.idf1_thresh
                << ",  MOTA " << kv.second.mota
                << " @ " << kv.second.mota_thresh << "\n";
    }
    std::cout << "\n";

    if( params.opt_filter_estimator != "none" )
    {
      success = write_dive_filter( sweep_dir + "/dive.config.json",
                                   sweep_scores,
                                   params.opt_filter_estimator ) && success;
    }
  }

  // The metrics JSON carries the confusion matrix, and optionally the curves,
  // so it needs the same pass the plot exports use. Generated once up front and
  // shared, rather than evaluated twice.
  bool need_plots = !params.opt_output_plots.empty() ||
                    !params.opt_output_pr_csv.empty() ||
                    !params.opt_output_conf_csv.empty() ||
                    !params.opt_output_roc_csv.empty() ||
                    !params.opt_output_metrics.empty();

  std::unique_ptr< viame::evaluation_plot_data > plot_data_ptr;

  if( need_plots )
  {
    try
    {
      LOG_INFO( g_logger, "Generating plot data..." );
      plot_data_ptr.reset(
        new viame::evaluation_plot_data( evaluator.generate_plot_data() ) );
    }
    catch( const std::exception& e )
    {
      LOG_ERROR( g_logger, "Failed to generate plot data: " << e.what() );
    }
  }

  if( !params.opt_output_metrics.empty() )
  {
    success = write_metrics_json( results, params.opt_output_metrics,
                                  plot_data_ptr.get(),
                                  params.opt_json_curves ) && success;
  }

  if( !params.opt_output_summary.empty() )
  {
    success = write_summary_text( results, params.opt_output_summary ) && success;
  }

  if( plot_data_ptr )
  {
    try
    {
      const auto& plot_data = *plot_data_ptr;

      // Export full plot data to directory
      if( !params.opt_output_plots.empty() )
      {
        // Create output directory if needed
        if( !kwiversys::SystemTools::FileIsDirectory( params.opt_output_plots ) &&
            !kwiversys::SystemTools::MakeDirectory( params.opt_output_plots ) )
        {
          LOG_ERROR( g_logger, "Could not create plot output directory: "
                     << params.opt_output_plots );
          return EXIT_FAILURE;
        }

        // Export CSV data files
        if( viame::model_evaluator::export_plot_data( plot_data, params.opt_output_plots ) )
        {
          LOG_INFO( g_logger, "Plot CSV data written to: " << params.opt_output_plots );
        }
        else
        {
          LOG_ERROR( g_logger, "Failed to export plot CSV data" );
          success = false;
        }

#ifdef VIAME_TOOLS_HAVE_OPENCV
        // Render plot images using OpenCV
        LOG_INFO( g_logger, "Rendering plot images..." );
        viame::metrics_plotter plotter;
        if( plotter.render_all_plots( plot_data, params.opt_output_plots ) )
        {
          LOG_INFO( g_logger, "Plot images rendered to: " << params.opt_output_plots );
        }
        else
        {
          LOG_WARN( g_logger, "Some plot images could not be rendered" );
        }
#else
        LOG_INFO( g_logger, "Plot images need an OpenCV-enabled build; "
          "wrote the plot data only" );
#endif
      }

      // Export individual plots
      if( !params.opt_output_pr_csv.empty() )
      {
        if( viame::model_evaluator::export_pr_curve_csv(
              plot_data.overall_pr_curve, params.opt_output_pr_csv ) )
        {
          LOG_INFO( g_logger, "PR curve written to: " << params.opt_output_pr_csv );
        }
        else
        {
          LOG_ERROR( g_logger, "Failed to export PR curve" );
          success = false;
        }
      }

      if( !params.opt_output_conf_csv.empty() )
      {
        if( viame::model_evaluator::export_confusion_matrix_csv(
              plot_data.confusion_matrix, params.opt_output_conf_csv ) )
        {
          LOG_INFO( g_logger, "Confusion matrix written to: " << params.opt_output_conf_csv );
        }
        else
        {
          LOG_ERROR( g_logger, "Failed to export confusion matrix" );
          success = false;
        }
      }

      if( !params.opt_output_roc_csv.empty() )
      {
        // Same schema as roc_curve_overall.csv in the plot directory
        std::ofstream out( params.opt_output_roc_csv );
        if( out.is_open() )
        {
          out << "confidence,false_alarms_per_frame,true_positive_rate\n";
          out << std::fixed << std::setprecision( 6 );

          for( const auto& pt : plot_data.overall_roc_curve.points )
          {
            // The leading anchor point sits above every detection's confidence
            if( std::isfinite( pt.confidence ) )
            {
              out << pt.confidence;
            }
            else
            {
              out << "inf";
            }

            out << "," << pt.false_alarms_per_frame
                << "," << pt.true_positive_rate << "\n";
          }

          out.close();
          LOG_INFO( g_logger, "ROC curve written to: " << params.opt_output_roc_csv );
        }
        else
        {
          LOG_ERROR( g_logger, "Failed to open ROC output file" );
          success = false;
        }
      }
    }
    catch( const std::exception& e )
    {
      LOG_ERROR( g_logger, "Plot generation failed: " << e.what() );
      return EXIT_FAILURE;
    }
  }

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace tools
} // namespace viame
