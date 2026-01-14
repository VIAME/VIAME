/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Command-line tool for scoring detection/tracking results using evaluate_models

#include <viame/evaluate_models.h>
#include <viame/plot_metrics.h>

#include <kwiversys/SystemTools.hxx>
#include <kwiversys/CommandLineArguments.hxx>
#include <kwiversys/Directory.hxx>

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/logger/logger.h>

#include <vector>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

// =============================================================================
// Global variables and parameter class
// =============================================================================

class score_results_params
{
public:
  kwiversys::CommandLineArguments m_args;

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
  unsigned opt_frame_tolerance = 0;

  // Output options
  std::string opt_output_metrics;    // Output metrics to JSON file
  std::string opt_output_summary;    // Output summary to text file
  std::string opt_output_plots;      // Output plot data to directory
  std::string opt_output_pr_csv;     // Output PR curve to CSV
  std::string opt_output_conf_csv;   // Output confusion matrix to CSV
  std::string opt_output_roc_csv;    // Output ROC curve to CSV
  bool opt_print_summary = true;

  score_results_params() = default;
  virtual ~score_results_params() = default;
};

static score_results_params g_params;
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
  for( const auto& c : computed )
  {
    std::string base = kwiversys::SystemTools::GetFilenameWithoutLastExtension( c );

    auto it = truth_map.find( base );
    if( it != truth_map.end() )
    {
      pairs.push_back( { c, it->second } );
    }
    else
    {
      LOG_WARN( g_logger, "No matching ground truth for: " << c );
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
            << std::setw( 12 ) << "AP"
            << "\n";
  std::cout << std::string( 103, '-' ) << "\n";

  for( const auto& kv : results.per_class_metrics )
  {
    const auto& class_name = kv.first;
    const auto& metrics = kv.second;

    auto get_metric = [&]( const std::string& name ) -> double
    {
      auto it = metrics.find( name );
      return ( it != metrics.end() ) ? it->second : 0.0;
    };

    std::cout << std::left << std::setw( 25 ) << class_name
              << std::right << std::setw( 10 ) << static_cast< int >( get_metric( "true_positives" ) )
              << std::setw( 10 ) << static_cast< int >( get_metric( "false_positives" ) )
              << std::setw( 10 ) << static_cast< int >( get_metric( "false_negatives" ) )
              << std::fixed << std::setprecision( 4 )
              << std::setw( 12 ) << get_metric( "precision" )
              << std::setw( 12 ) << get_metric( "recall" )
              << std::setw( 12 ) << get_metric( "f1_score" )
              << std::setw( 12 ) << get_metric( "average_precision" )
              << "\n";
  }
  std::cout << "\n";
}

bool write_metrics_json( const viame::evaluation_results& results,
                         const std::string& filepath )
{
  std::ofstream out( filepath );
  if( !out.is_open() )
  {
    LOG_ERROR( g_logger, "Could not open output file: " << filepath );
    return false;
  }

  out << std::fixed << std::setprecision( 6 );
  out << "{\n";

  bool first = true;
  for( const auto& kv : results.all_metrics )
  {
    if( !first ) out << ",\n";
    out << "  \"" << escape_json( kv.first ) << "\": " << kv.second;
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
        out << "\"" << escape_json( metric_kv.first ) << "\": " << metric_kv.second;
        first_metric = false;
      }

      out << "}";
      first_class = false;
    }

    out << "\n  }";
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

int main( int argc, char* argv[] )
{
  // Initialize logger
  g_logger = kwiver::vital::get_logger( "viame.tools.score_results" );

  // Setup command line arguments
  typedef kwiversys::CommandLineArguments argT;

  g_params.m_args.Initialize( argc, argv );

  // General options
  g_params.m_args.AddArgument( "--help", argT::NO_ARGUMENT,
    &g_params.opt_help, "Display usage information" );
  g_params.m_args.AddArgument( "-h", argT::NO_ARGUMENT,
    &g_params.opt_help, "Display usage information" );
  g_params.m_args.AddArgument( "--verbose", argT::NO_ARGUMENT,
    &g_params.opt_verbose, "Enable verbose output" );
  g_params.m_args.AddArgument( "-v", argT::NO_ARGUMENT,
    &g_params.opt_verbose, "Enable verbose output" );

  // Input options
  g_params.m_args.AddArgument( "--computed", argT::SPACE_ARGUMENT,
    &g_params.opt_computed, "Input computed detection/track file or folder" );
  g_params.m_args.AddArgument( "-c", argT::SPACE_ARGUMENT,
    &g_params.opt_computed, "Input computed detection/track file or folder" );
  g_params.m_args.AddArgument( "--truth", argT::SPACE_ARGUMENT,
    &g_params.opt_truth, "Input ground truth file or folder" );
  g_params.m_args.AddArgument( "-t", argT::SPACE_ARGUMENT,
    &g_params.opt_truth, "Input ground truth file or folder" );
  g_params.m_args.AddArgument( "--input-ext", argT::SPACE_ARGUMENT,
    &g_params.opt_input_ext, "File extension filter for folder inputs (default: .csv)" );

  // Scoring options
  g_params.m_args.AddArgument( "--iou-threshold", argT::SPACE_ARGUMENT,
    &g_params.opt_iou_threshold, "IoU threshold for matching (default: 0.5)" );
  g_params.m_args.AddArgument( "--iou", argT::SPACE_ARGUMENT,
    &g_params.opt_iou_threshold, "IoU threshold for matching (default: 0.5)" );
  g_params.m_args.AddArgument( "--confidence-threshold", argT::SPACE_ARGUMENT,
    &g_params.opt_confidence_threshold, "Minimum confidence threshold (default: 0.0)" );
  g_params.m_args.AddArgument( "--conf", argT::SPACE_ARGUMENT,
    &g_params.opt_confidence_threshold, "Minimum confidence threshold (default: 0.0)" );
  g_params.m_args.AddArgument( "--per-class", argT::NO_ARGUMENT,
    &g_params.opt_per_class, "Compute per-class metrics" );
  g_params.m_args.AddArgument( "--no-tracking", argT::NO_ARGUMENT,
    &g_params.opt_compute_tracking, "Disable tracking metrics computation" );
  g_params.m_args.AddArgument( "--frame-tolerance", argT::SPACE_ARGUMENT,
    &g_params.opt_frame_tolerance, "Frame tolerance for temporal matching (default: 0)" );

  // Output options
  g_params.m_args.AddArgument( "--output-metrics", argT::SPACE_ARGUMENT,
    &g_params.opt_output_metrics, "Output all metrics to JSON file" );
  g_params.m_args.AddArgument( "-o", argT::SPACE_ARGUMENT,
    &g_params.opt_output_metrics, "Output all metrics to JSON file" );
  g_params.m_args.AddArgument( "--output-summary", argT::SPACE_ARGUMENT,
    &g_params.opt_output_summary, "Output summary to text file" );
  g_params.m_args.AddArgument( "--output-plots", argT::SPACE_ARGUMENT,
    &g_params.opt_output_plots, "Output plot data (PR, ROC, confusion) to directory" );
  g_params.m_args.AddArgument( "--output-pr-csv", argT::SPACE_ARGUMENT,
    &g_params.opt_output_pr_csv, "Output precision-recall curve to CSV" );
  g_params.m_args.AddArgument( "--output-conf-csv", argT::SPACE_ARGUMENT,
    &g_params.opt_output_conf_csv, "Output confusion matrix to CSV" );
  g_params.m_args.AddArgument( "--output-roc-csv", argT::SPACE_ARGUMENT,
    &g_params.opt_output_roc_csv, "Output ROC curve to CSV" );
  g_params.m_args.AddArgument( "--no-print", argT::NO_ARGUMENT,
    &g_params.opt_print_summary, "Suppress printing summary to stdout" );

  // Parse command line
  if( !g_params.m_args.Parse() )
  {
    LOG_ERROR( g_logger, "Problem parsing arguments" );
    return EXIT_FAILURE;
  }

  // Handle inverted boolean flags
  for( int i = 1; i < argc; ++i )
  {
    std::string arg = argv[i];
    if( arg == "--no-tracking" )
    {
      g_params.opt_compute_tracking = false;
    }
    else if( arg == "--no-print" )
    {
      g_params.opt_print_summary = false;
    }
  }

  // Display help
  if( g_params.opt_help )
  {
    std::cout << "Usage: " << argv[0] << " [options]\n\n"
              << "Score detection and tracking results using the evaluate_models library.\n"
              << "Computes comprehensive metrics including precision, recall, F1, AP,\n"
              << "MOT metrics (MOTA, MOTP, IDF1), HOTA, and KWANT-style metrics.\n\n"
              << "Options:\n"
              << g_params.m_args.GetHelp()
              << "\nExamples:\n"
              << "  " << argv[0] << " -c detections.csv -t groundtruth.csv\n"
              << "  " << argv[0] << " -c results/ -t truth/ --iou 0.5 --per-class\n"
              << "  " << argv[0] << " -c det.csv -t gt.csv -o metrics.json --output-plots plots/\n"
              << "  " << argv[0] << " -c det.csv -t gt.csv --output-pr-csv pr_curve.csv\n"
              << std::endl;
    return EXIT_SUCCESS;
  }

  // Validate inputs
  if( g_params.opt_computed.empty() )
  {
    LOG_ERROR( g_logger, "No computed file/folder specified. Use --computed or -c option." );
    return EXIT_FAILURE;
  }

  if( g_params.opt_truth.empty() )
  {
    LOG_ERROR( g_logger, "No ground truth file/folder specified. Use --truth or -t option." );
    return EXIT_FAILURE;
  }

  if( !kwiversys::SystemTools::FileExists( g_params.opt_computed ) )
  {
    LOG_ERROR( g_logger, "Computed path does not exist: " << g_params.opt_computed );
    return EXIT_FAILURE;
  }

  if( !kwiversys::SystemTools::FileExists( g_params.opt_truth ) )
  {
    LOG_ERROR( g_logger, "Ground truth path does not exist: " << g_params.opt_truth );
    return EXIT_FAILURE;
  }

  // Set default extension
  if( g_params.opt_input_ext.empty() )
  {
    g_params.opt_input_ext = ".csv";
  }
  else if( g_params.opt_input_ext[0] != '.' )
  {
    g_params.opt_input_ext = "." + g_params.opt_input_ext;
  }

  // Load plugins (needed for CSV readers)
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  // Collect input files
  auto computed_files = collect_files( g_params.opt_computed, g_params.opt_input_ext );
  auto truth_files = collect_files( g_params.opt_truth, g_params.opt_input_ext );

  if( computed_files.empty() )
  {
    LOG_ERROR( g_logger, "No computed files found in: " << g_params.opt_computed );
    return EXIT_FAILURE;
  }

  if( truth_files.empty() )
  {
    LOG_ERROR( g_logger, "No ground truth files found in: " << g_params.opt_truth );
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

  if( g_params.opt_verbose )
  {
    for( size_t i = 0; i < final_computed.size(); ++i )
    {
      LOG_INFO( g_logger, "  " << final_computed[i] << " <-> " << final_truth[i] );
    }
  }

  // Configure evaluation
  viame::evaluation_config config;
  config.iou_threshold = g_params.opt_iou_threshold;
  config.confidence_threshold = g_params.opt_confidence_threshold;
  config.compute_tracking_metrics = g_params.opt_compute_tracking;
  config.compute_per_class_metrics = g_params.opt_per_class;
  config.frame_tolerance = g_params.opt_frame_tolerance;

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
  if( g_params.opt_print_summary )
  {
    print_summary( results );
    if( g_params.opt_per_class )
    {
      print_per_class_metrics( results );
    }
  }

  // Write outputs
  bool success = true;

  if( !g_params.opt_output_metrics.empty() )
  {
    success = write_metrics_json( results, g_params.opt_output_metrics ) && success;
  }

  if( !g_params.opt_output_summary.empty() )
  {
    success = write_summary_text( results, g_params.opt_output_summary ) && success;
  }

  // Generate and export plot data if requested
  bool need_plots = !g_params.opt_output_plots.empty() ||
                    !g_params.opt_output_pr_csv.empty() ||
                    !g_params.opt_output_conf_csv.empty() ||
                    !g_params.opt_output_roc_csv.empty();

  if( need_plots )
  {
    LOG_INFO( g_logger, "Generating plot data..." );

    // Export full plot data to directory
    if( !g_params.opt_output_plots.empty() )
    {
      // Create output directory if needed
      if( !kwiversys::SystemTools::FileIsDirectory( g_params.opt_output_plots ) )
      {
        kwiversys::SystemTools::MakeDirectory( g_params.opt_output_plots );
      }

      auto plot_data = evaluator.generate_plot_data();

      // Export CSV data files
      if( viame::model_evaluator::export_plot_data( plot_data, g_params.opt_output_plots ) )
      {
        LOG_INFO( g_logger, "Plot CSV data written to: " << g_params.opt_output_plots );
      }
      else
      {
        LOG_ERROR( g_logger, "Failed to export plot CSV data" );
        success = false;
      }

      // Render plot images using OpenCV
      LOG_INFO( g_logger, "Rendering plot images..." );
      viame::metrics_plotter plotter;
      if( plotter.render_all_plots( plot_data, g_params.opt_output_plots ) )
      {
        LOG_INFO( g_logger, "Plot images rendered to: " << g_params.opt_output_plots );
      }
      else
      {
        LOG_WARN( g_logger, "Some plot images could not be rendered" );
      }
    }

    // Export individual plots
    if( !g_params.opt_output_pr_csv.empty() )
    {
      auto pr_curve = evaluator.generate_pr_curve();
      if( viame::model_evaluator::export_pr_curve_csv( pr_curve, g_params.opt_output_pr_csv ) )
      {
        LOG_INFO( g_logger, "PR curve written to: " << g_params.opt_output_pr_csv );
      }
      else
      {
        LOG_ERROR( g_logger, "Failed to export PR curve" );
        success = false;
      }
    }

    if( !g_params.opt_output_conf_csv.empty() )
    {
      auto conf_matrix = evaluator.generate_confusion_matrix();
      if( viame::model_evaluator::export_confusion_matrix_csv( conf_matrix, g_params.opt_output_conf_csv ) )
      {
        LOG_INFO( g_logger, "Confusion matrix written to: " << g_params.opt_output_conf_csv );
      }
      else
      {
        LOG_ERROR( g_logger, "Failed to export confusion matrix" );
        success = false;
      }
    }

    if( !g_params.opt_output_roc_csv.empty() )
    {
      auto plot_data = evaluator.generate_plot_data();

      // Write ROC curve manually since there's no dedicated export function
      std::ofstream out( g_params.opt_output_roc_csv );
      if( out.is_open() )
      {
        out << "false_positive_rate,true_positive_rate,confidence\n";
        for( const auto& pt : plot_data.overall_roc_curve.points )
        {
          out << pt.false_positive_rate << ","
              << pt.true_positive_rate << ","
              << pt.confidence << "\n";
        }
        out.close();
        LOG_INFO( g_logger, "ROC curve written to: " << g_params.opt_output_roc_csv );
      }
      else
      {
        LOG_ERROR( g_logger, "Failed to open ROC output file" );
        success = false;
      }
    }
  }

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
