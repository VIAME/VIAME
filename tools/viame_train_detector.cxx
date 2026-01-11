/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <kwiversys/SystemTools.hxx>
#include <kwiversys/CommandLineArguments.hxx>

#include <vital/kwiver-include-paths.h>

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/config/config_block.h>
#include <vital/config/config_block_io.h>
#include <vital/util/demangle.h>
#include <vital/util/wrap_text_block.h>
#include <vital/algo/algorithm_factory.h>
#include <vital/algo/train_detector.h>
#include <vital/algo/train_tracker.h>
#include <vital/algo/detected_object_set_input.h>
#include <vital/algo/read_object_track_set.h>
#include <vital/algo/image_io.h>
#include <vital/types/image_container.h>
#include <vital/types/object_track_set.h>
#include <vital/logger/logger.h>

#include <sprokit/pipeline/process_exception.h>

#include "utilities_file.h"
#include "utilities_training.h"

#include <vector>
#include <unordered_set>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <memory>
#include <cctype>
#include <regex>

#if WIN32 || ( __cplusplus >= 201703L && __has_include(<filesystem>) )
  #include <filesystem>
  namespace filesystem = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace filesystem = std::experimental::filesystem;
#endif

using namespace viame;

// =======================================================================================
// Class storing all input parameters and private variables for tool
class trainer_vars
{
public:

  // Collected command line args
  kwiversys::CommandLineArguments m_args;

  // Config options
  bool opt_help;
  bool opt_list;
  bool opt_no_query;
  bool opt_no_adv_print;
  bool opt_no_emb_pipe;
  bool opt_gt_only;

  std::string opt_config;
  std::string opt_input_dir;
  std::string opt_input_list;
  std::string opt_input_truth;
  std::string opt_label_file;
  std::string opt_validation_dir;
  std::string opt_detector;
  std::string opt_tracker;
  std::string opt_out_config;
  std::string opt_threshold;
  std::string opt_settings;
  std::string opt_pipeline_file;
  std::string opt_frame_rate;
  std::string opt_max_frame_count;
  std::string opt_timeout;
  std::string opt_init_weights;

  trainer_vars()
  {
    opt_help = false;
    opt_list = false;
    opt_no_query = false;
    opt_no_adv_print = false;
    opt_no_emb_pipe = false;
    opt_gt_only = false;
  }

  virtual ~trainer_vars()
  {
  }
};

// =======================================================================================
// Define global variables used across this tool
static trainer_vars g_params;
static kwiver::vital::logger_handle_t g_logger;


// =======================================================================================
// Assorted configuration related helper functions
static kwiver::vital::config_block_sptr default_config()
{
  kwiver::vital::config_block_sptr config
    = kwiver::vital::config_block::empty_config( "detector_trainer_tool" );

  config->set_value( "groundtruth_extensions", ".csv",
    "Groundtruth file extensions (csv, kw18, txt, etc...). Note: this is independent of "
    "the format that's stored in the file" );
  config->set_value( "groundtruth_style", "one_per_folder",
    "Can be either: \"one_per_file\" or \"one_per_folder\"" );
  config->set_value( "augmentation_pipeline", "",
    "Optional embedded pipeline for performing assorted augmentations" );
  config->set_value( "augmentation_cache", "augmented_images",
    "Directory to store augmented samples, a temp directiry is used if not specified." );
  config->set_value( "regenerate_cache", "true",
    "If an augmentation cache already exists, should we regenerate it or use it as-is?" );
  config->set_value( "augmented_ext_override", ".png",
    "Optional image extension over-ride for augmented images." );
  config->set_value( "default_percent_validation", "0.05",
    "Percent [0.0, 1.0] of validation samples to use if no manual files specified." );
  config->set_value( "validation_burst_frame_count", "500",
    "Number of sequential frames to use in validation set to avoid it being too similar to "
    "the training set." );
  config->set_value( "image_extensions",
    ".jpg;.jpeg;.JPG;.JPEG;.tif;.tiff;.TIF;.TIFF;.png;.PNG;.sgi;.SGI;.bmp;.BMP;.pgm;.PGM",
    "Semicolon list of seperated image extensions to use in training, images without "
    "this extension will not be included." );
  config->set_value( "video_extensions",
    ".mp4;.MP4;.mpg;.MPG;.mpeg;.MPEG;.avi;.AVI;.wmv;.WMV;.mov;.MOV;.webm;.WEBM;.ogg;.OGG",
    "Semicolon list of seperated video extensions to use in training, images without "
    "this extension will not be included." );
  config->set_value( "video_extractor", "ffmpeg",
    "Method to use to extract frames from video, can either be ffmpeg or a pipe file" );
  config->set_value( "frame_rate", "5",
    "Default frame rate to use for videos when it is not manually specified inside of a "
    "groundtruth file." );
  config->set_value( "threshold", "0.00",
    "Optional threshold to provide on top of input groundtruth. This is useful if the "
    "truth is derived from some automated detector and is unfiltered." );
  config->set_value( "max_frame_count", "0",
    "Maximum number of frames to use in training, useful for debugging and speed "
    "optimization purposes." );
  config->set_value( "use_labels", "true",
    "Adjust labels based on labels.txt file in this loader instead of passing the "
    "responsibility to individual detector trainer classes." );
  config->set_value( "downsample", "0",
    "Downsample factor applied across all inputs." );
  config->set_value( "targetted_downsample", "0",
    "Extra downsample factor applied across certain inputs." );
  config->set_value( "targetted_downsample_string", "",
    "Apply extra targetted downsample to inputs containing this string." );
  config->set_value( "secondary_frame_labels", "",
    "Secondary categories should be suppressed if a primary category is present "
    "on a particular frame." );
  config->set_value( "ignore_secondary_burst_count", "0",
    "After receiving a frame with a primary category, ignore this many secondary "
    "category frames before using the next secondary frame." );
  config->set_value( "secondary_downsample", "0",
    "Downsample factor for frames containing only secondary classes." );
  config->set_value( "check_override", "false",
    "Over-ride and ignore data safety checks." );
  config->set_value( "convert_to_full_frame", "false",
    "Convert input detections to full frame labels even if they're not." );
  config->set_value( "data_warning_file", "",
    "Optional file for storing possible data errors and warning." );

  kwiver::vital::algo::detected_object_set_input::get_nested_algo_configuration
    ( "groundtruth_reader", config, kwiver::vital::algo::detected_object_set_input_sptr() );
  kwiver::vital::algo::image_io::get_nested_algo_configuration
    ( "image_reader", config, kwiver::vital::algo::image_io_sptr() );
  kwiver::vital::algo::train_detector::get_nested_algo_configuration
    ( "detector_trainer", config, kwiver::vital::algo::train_detector_sptr() );
  kwiver::vital::algo::train_tracker::get_nested_algo_configuration
    ( "tracker_trainer", config, kwiver::vital::algo::train_tracker_sptr() );
  kwiver::vital::algo::read_object_track_set::get_nested_algo_configuration
    ( "track_reader", config, kwiver::vital::algo::read_object_track_set_sptr() );

  return config;
}

// =======================================================================================
/*                   _
 *   _ __ ___   __ _(_)_ __
 *  | '_ ` _ \ / _` | | '_ \
 *  | | | | | | (_| | | | | |
 *  |_| |_| |_|\__,_|_|_| |_|
 *
 */
int
main( int argc, char* argv[] )
{
  // Initialize shared storage
  g_logger = kwiver::vital::get_logger( "viame_train_detector" );

  // Parse options
  g_params.m_args.Initialize( argc, argv );
  g_params.m_args.StoreUnusedArguments( true );
  typedef kwiversys::CommandLineArguments argT;

  g_params.m_args.AddArgument( "--help",          argT::NO_ARGUMENT,
    &g_params.opt_help, "Display usage information" );
  g_params.m_args.AddArgument( "-h",              argT::NO_ARGUMENT,
    &g_params.opt_help, "Display usage information" );
  g_params.m_args.AddArgument( "--list",          argT::NO_ARGUMENT,
    &g_params.opt_list, "Display list of all trainable algorithms" );
  g_params.m_args.AddArgument( "-l",              argT::NO_ARGUMENT,
    &g_params.opt_list, "Display list of all trainable algorithms" );
  g_params.m_args.AddArgument( "--no-query",      argT::NO_ARGUMENT,
    &g_params.opt_no_query, "Do not query the user for anything" );
  g_params.m_args.AddArgument( "-nq",             argT::NO_ARGUMENT,
    &g_params.opt_no_query, "Do not query the user for anything" );
  g_params.m_args.AddArgument( "--no-adv-prints", argT::NO_ARGUMENT,
    &g_params.opt_no_adv_print, "Do not print out any advanced chars" );
  g_params.m_args.AddArgument( "-nap",            argT::NO_ARGUMENT,
    &g_params.opt_no_adv_print, "Do not print out any advanced chars" );
  g_params.m_args.AddArgument( "--no-embedded-pipe", argT::NO_ARGUMENT,
    &g_params.opt_no_emb_pipe, "Do not output embedded pipes" );
  g_params.m_args.AddArgument( "-nep",            argT::NO_ARGUMENT,
    &g_params.opt_no_emb_pipe, "Do not output embedded pipes" );
  g_params.m_args.AddArgument( "--gt-frames-only", argT::NO_ARGUMENT,
    &g_params.opt_gt_only, "Use frames with annotations only" );
  g_params.m_args.AddArgument( "-gto",            argT::NO_ARGUMENT,
    &g_params.opt_gt_only, "Use frames with annotations only" );
  g_params.m_args.AddArgument( "--config",        argT::SPACE_ARGUMENT,
    &g_params.opt_config, "Input configuration file(s) with parameters" );
  g_params.m_args.AddArgument( "-c",              argT::SPACE_ARGUMENT,
    &g_params.opt_config, "Input configuration file(s) with parameters" );
  g_params.m_args.AddArgument( "--input",         argT::SPACE_ARGUMENT,
    &g_params.opt_input_dir, "Input directory containing groundtruth" );
  g_params.m_args.AddArgument( "-i",              argT::SPACE_ARGUMENT,
    &g_params.opt_input_dir, "Input directory containing groundtruth" );
  g_params.m_args.AddArgument( "--input-list",    argT::SPACE_ARGUMENT,
    &g_params.opt_input_list, "Input list with data for training" );
  g_params.m_args.AddArgument( "-il",             argT::SPACE_ARGUMENT,
    &g_params.opt_input_list, "Input list with data for training" );
  g_params.m_args.AddArgument( "--input-truth",   argT::SPACE_ARGUMENT,
    &g_params.opt_input_truth, "Input list containing training truth" );
  g_params.m_args.AddArgument( "-it",             argT::SPACE_ARGUMENT,
    &g_params.opt_input_truth, "Input list containing training truth" );
  g_params.m_args.AddArgument( "--labels",        argT::SPACE_ARGUMENT,
    &g_params.opt_label_file, "Input label file for train categories" );
  g_params.m_args.AddArgument( "-lbl",            argT::SPACE_ARGUMENT,
    &g_params.opt_label_file, "Input label file for train categories" );
  g_params.m_args.AddArgument( "--validation",    argT::SPACE_ARGUMENT,
    &g_params.opt_validation_dir, "Optional validation input directory" );
  g_params.m_args.AddArgument( "-v",              argT::SPACE_ARGUMENT,
    &g_params.opt_validation_dir, "Optional validation input directory" );
  g_params.m_args.AddArgument( "--detector",      argT::SPACE_ARGUMENT,
    &g_params.opt_detector, "Type of detector(s) to train if no config" );
  g_params.m_args.AddArgument( "-d",              argT::SPACE_ARGUMENT,
    &g_params.opt_detector, "Type of detector(s) to train if no config" );
  g_params.m_args.AddArgument( "--tracker",       argT::SPACE_ARGUMENT,
    &g_params.opt_tracker, "Type of tracker(s) to train (optional)" );
  g_params.m_args.AddArgument( "-tt",             argT::SPACE_ARGUMENT,
    &g_params.opt_tracker, "Type of tracker(s) to train (optional)" );
  g_params.m_args.AddArgument( "--output-config", argT::SPACE_ARGUMENT,
    &g_params.opt_out_config, "Output a sample configuration to file" );
  g_params.m_args.AddArgument( "-o",              argT::SPACE_ARGUMENT,
    &g_params.opt_out_config, "Output a sample configuration to file" );
  g_params.m_args.AddArgument( "--setting",       argT::SPACE_ARGUMENT,
    &g_params.opt_settings, "Over-ride some setting in the config" );
  g_params.m_args.AddArgument( "-s",              argT::SPACE_ARGUMENT,
    &g_params.opt_settings, "Over-ride some setting in the config" );
  g_params.m_args.AddArgument( "--threshold",     argT::SPACE_ARGUMENT,
    &g_params.opt_threshold, "Threshold override to apply over input" );
  g_params.m_args.AddArgument( "-t",              argT::SPACE_ARGUMENT,
    &g_params.opt_threshold, "Threshold override to apply over input" );
  g_params.m_args.AddArgument( "--pipeline",      argT::SPACE_ARGUMENT,
    &g_params.opt_pipeline_file, "Pipeline file" );
  g_params.m_args.AddArgument( "-p",              argT::SPACE_ARGUMENT,
    &g_params.opt_pipeline_file, "Pipeline file" );
  g_params.m_args.AddArgument( "--default-vfr",   argT::SPACE_ARGUMENT,
    &g_params.opt_frame_rate, "Default video frame rate for extraction" );
  g_params.m_args.AddArgument( "-vfr",            argT::SPACE_ARGUMENT,
    &g_params.opt_frame_rate, "Default video frame rate for extraction" );
  g_params.m_args.AddArgument( "--max-frame-count",argT::SPACE_ARGUMENT,
    &g_params.opt_max_frame_count, "Maximum frame count to use" );
  g_params.m_args.AddArgument( "-mfc",            argT::SPACE_ARGUMENT,
    &g_params.opt_max_frame_count, "Maximum frame count to use" );
  g_params.m_args.AddArgument( "--timeout",       argT::SPACE_ARGUMENT,
    &g_params.opt_timeout, "Maximum time in seconds" );
  g_params.m_args.AddArgument( "-to",             argT::SPACE_ARGUMENT,
    &g_params.opt_timeout, "Maximum time in seconds" );
  g_params.m_args.AddArgument( "--init-weights",  argT::SPACE_ARGUMENT,
    &g_params.opt_init_weights, "Optional input seed weights over-ride" );
  g_params.m_args.AddArgument( "-iw",             argT::SPACE_ARGUMENT,
    &g_params.opt_init_weights, "Optional input seed weights over-ride" );

  // Parse args
  if( !g_params.m_args.Parse() )
  {
    std::cerr << "Problem parsing arguments" << std::endl;
    return EXIT_FAILURE;
  }

  // Print help
  if( argc == 1 || g_params.opt_help )
  {
    std::cout << "Usage: " << argv[0] << "[options]\n"
              << "\nTrain one of several object detectors in the system.\n"
              << g_params.m_args.GetHelp() << std::endl;
    return EXIT_FAILURE;
  }

  // List option
  if( g_params.opt_list )
  {
    kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

    kwiver::vital::path_list_t pathl;
    const std::string& default_module_paths( DEFAULT_MODULE_PATHS );

    kwiversys::SystemTools::Split( default_module_paths, pathl, PATH_SEPARATOR_CHAR );

    for( auto path : pathl )
    {
      vpm.add_search_path( path );
    }

    vpm.load_plugins( pathl );

    auto fact_list = vpm.get_factories( "train_detector" );

    if( fact_list.empty() )
    {
      std::cerr << "No loaded detectors to list" << std::endl;
    }
    else
    {
      std::cout << std::endl << "Trainable detector variants:" << std::endl << std::endl;
    }

    for( auto fact : fact_list )
    {
      std::string name;
      if( fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, name ) )
      {
        std::cout << name << std::endl;
      }
    }
    return EXIT_FAILURE;
  }

  // Test for presence of conflicting options
  if( !g_params.opt_config.empty() && !g_params.opt_detector.empty() )
  {
    std::cerr << "Only one of --config and --detector allowed." << std::endl;
    return EXIT_FAILURE;
  }

  // Test for presence of required options (either detector or tracker training)
  if( g_params.opt_config.empty() && g_params.opt_detector.empty() &&
      g_params.opt_tracker.empty() )
  {
    std::cerr << "One of --config, --detector, or --tracker must be set." << std::endl;
    return EXIT_FAILURE;
  }

  // Parse comma-separated configs or detectors/trackers for multi-model training
  std::vector< std::string > training_configs;
  std::vector< std::string > training_detectors;
  std::vector< std::string > training_trackers;

  if( !g_params.opt_config.empty() )
  {
    string_to_vector( g_params.opt_config, training_configs, "," );
  }
  if( !g_params.opt_detector.empty() )
  {
    string_to_vector( g_params.opt_detector, training_detectors, "," );
  }
  if( !g_params.opt_tracker.empty() )
  {
    string_to_vector( g_params.opt_tracker, training_trackers, "," );
  }

  const bool multi_model_training =
    ( training_configs.size() > 1 || training_detectors.size() > 1 );
  const unsigned model_count =
    std::max( training_configs.size(), training_detectors.size() );
  const bool train_trackers = !training_trackers.empty();

  if( multi_model_training )
  {
    std::cout << "Multi-model training enabled: " << model_count
              << " models will be trained sequentially" << std::endl;
  }

  // Load KWIVER plugins
  kwiver::vital::plugin_manager::instance().load_all_plugins();
  kwiver::vital::config_block_sptr config = default_config();
  kwiver::vital::algo::detected_object_set_input_sptr groundtruth_reader;
  kwiver::vital::algo::image_io_sptr image_reader;
  kwiver::vital::algo::train_detector_sptr detector_trainer;
  kwiver::vital::algo::train_tracker_sptr tracker_trainer;
  kwiver::vital::algo::read_object_track_set_sptr track_reader;

  // Read all configuration options and check settings (use first config/detector for data loading)
  std::string first_config = training_configs.empty() ? "" : training_configs[0];
  std::string first_detector = training_detectors.empty() ? "" : training_detectors[0];

  if( !first_config.empty() )
  {
    try
    {
      config->merge_config( kwiver::vital::read_config_file( first_config ) );
    }
    catch( const std::exception& e )
    {
      std::cerr << "Received exception: " << e.what() << std::endl
                << "Unable to load configuration file: "
                << first_config << std::endl;

      return EXIT_FAILURE;
    }
  }
  else
  {
    config->set_value( "detector_trainer:type", first_detector );
  }

  if( !g_params.opt_settings.empty() )
  {
    const std::string& setting = g_params.opt_settings;
    size_t const split_pos = setting.find( "=" );

    if( split_pos == std::string::npos )
    {
      std::string const reason = "Error: The setting on the command line \'"
        + setting + "\' does not contain the \'=\' string which separates "
        "the key from the value";

      throw std::runtime_error( reason );
    }

    kwiver::vital::config_block_key_t setting_key =
      setting.substr( 0, split_pos );
    kwiver::vital::config_block_value_t setting_value =
      setting.substr( split_pos + 1 );

    kwiver::vital::config_block_keys_t keys;

    kwiver::vital::tokenize( setting_key, keys,
      kwiver::vital::config_block::block_sep(),
      kwiver::vital::TokenizeTrimEmpty );

    if( keys.size() < 2 )
    {
      std::string const reason = "Error: The key portion of setting "
        "\'" + setting + "\' does not contain at least two keys in its "
        "keypath which is invalid. (e.g. must be at least a:b)";
  
      throw std::runtime_error( reason );
    }

    config->set_value( setting_key, setting_value );
  }

  if( g_params.opt_no_adv_print )
  {
    const std::string prefix1 = "detector_trainer:netharn";
    const std::string prefix2 = "detector_trainer:ocv_windowed:trainer:netharn";

    config->set_value( prefix1 + ":allow_unicode", "False" );
    config->set_value( prefix2 + ":allow_unicode", "False" );
  }

  // No need for conf_values to be in scope for rest of func taking up stack
  {
    auto conf_values = config->available_values();

    for( auto conf : conf_values )
    {
      if( conf.find( "timeout" ) != std::string::npos )
      {
        if( config->get_value< std::string >( conf ) == "default" ||
            !g_params.opt_timeout.empty() )
        {
          if( !g_params.opt_timeout.empty() )
          {
            config->set_value( conf, g_params.opt_timeout );
          }
          else
          {
            config->set_value( conf, "1209600" );
          }
        }
      }
    }

    if( g_params.opt_no_emb_pipe )
    {
      for( auto conf : conf_values )
      {
        if( conf.find( "pipeline_template" ) != std::string::npos )
        {
          std::string new_value = std::regex_replace(
            config->get_value< std::string >( conf ),
            std::regex( "embedded_" ),
            "detector_" );

          config->set_value( conf, new_value );
        }
      }
    }
  }

  if( !g_params.opt_init_weights.empty() )
  {
    bool any_config_found = false;
    auto conf_values = config->available_values();

    std::map< std::string, std::vector< std::string > > weight_ext =
      {
        { ".zip", { "seed_model" } },
        { ".pth", { "backbone", "seed_weights" } },
        { ".pt", { "backbone", "seed_weights" } },
        { ".py", { "config" } },
        { ".weights", { "seed_weights" } },
        { ".wt", { "seed_weights" } }
      };

    std::map< std::string, std::string > found_files;

    if( does_folder_exist( g_params.opt_init_weights ) )
    {
      for( auto itr : weight_ext )
      {
        std::vector< std::string > files_of_ext;

        list_files_in_folder( g_params.opt_init_weights,
                              files_of_ext,
                              false,
                              { itr.first } );

        if( files_of_ext.size() == 1 )
        {
          found_files[ itr.first ] = files_of_ext[0];
        }
        else if( files_of_ext.size() > 1 )
        {
          std::cout << "Folder contains multiple files of type "
                    << itr.first << std::endl;

          return EXIT_FAILURE;
        }
      }
    }
    else if( does_file_exist( g_params.opt_init_weights ) )
    {
      for( auto ext_itr : weight_ext )
      {
        if( ends_with_extension( g_params.opt_init_weights, ext_itr.first ) )
        {
          found_files[ ext_itr.first ] = g_params.opt_init_weights;
          break;
        }
      }
    }
    else
    {
      std::cout << "Seed weight path does not exist." << std::endl;
      return EXIT_FAILURE;
    }

    if( found_files.empty() )
    {
      std::cout << "Seed weight file or folder is of unknown type." << std::endl;
      return EXIT_FAILURE;
    }

    for( auto conf : conf_values )
    {
      for( auto found_itr : found_files )
      {
        for( auto search_str : weight_ext[ found_itr.first ] )
        {
          if( conf.find( search_str ) != std::string::npos )
          {
            config->set_value( conf, found_itr.second );
            any_config_found = true;
          }
        }
      }
    }

    if( !any_config_found )
    {
      std::cout << "Input seed weights are from a different model type." << std::endl;
      return EXIT_FAILURE;
    }
  }

  kwiver::vital::algo::train_detector::set_nested_algo_configuration
    ( "detector_trainer", config, detector_trainer );
  kwiver::vital::algo::train_detector::get_nested_algo_configuration
    ( "detector_trainer", config, detector_trainer );

  kwiver::vital::algo::detected_object_set_input::set_nested_algo_configuration
    ( "groundtruth_reader", config, groundtruth_reader );
  kwiver::vital::algo::detected_object_set_input::get_nested_algo_configuration
    ( "groundtruth_reader", config, groundtruth_reader );

  bool valid_config = true;

  if( !kwiver::vital::algo::detected_object_set_input::
        check_nested_algo_configuration( "groundtruth_reader", config ) )
  {
    valid_config = false;
  }

  if( !kwiver::vital::algo::train_detector::
        check_nested_algo_configuration( "detector_trainer", config ) )
  {
    valid_config = false;
  }

  if( !g_params.opt_out_config.empty() )
  {
    write_config_file( config, g_params.opt_out_config );

    if( valid_config )
    {
      std::cout << "Configuration file contained valid parameters "
        "and may be used for running" << std::endl;
      return EXIT_SUCCESS;
    }
    else
    {
      std::cout << "Configuration deemed not valid." << std::endl;
      return EXIT_FAILURE;
    }
  }
  else if( !valid_config )
  {
    std::cout << "Configuration not valid." << std::endl;
    return EXIT_FAILURE;
  }

  // Read setup configs
  std::string groundtruth_exts_str =
    config->get_value< std::string >( "groundtruth_extensions" );
  std::string groundtruth_style =
    config->get_value< std::string >( "groundtruth_style" );
  std::string pipeline_file =
    config->get_value< std::string >( "augmentation_pipeline" );
  std::string augmented_cache =
    config->get_value< std::string >( "augmentation_cache" );
  bool regenerate_cache =
    config->get_value< bool >( "regenerate_cache" );
  std::string augmented_ext_override =
    config->get_value< std::string >( "augmented_ext_override" );
  double percent_validation =
    config->get_value< double >( "default_percent_validation" );
  unsigned validation_burst_frame_count =
    config->get_value< unsigned >( "validation_burst_frame_count" );
  std::string image_exts_str =
    config->get_value< std::string >( "image_extensions" );
  std::string video_exts_str =
    config->get_value< std::string >( "video_extensions" );
  std::string video_extractor =
    config->get_value< std::string >( "video_extractor" );
  double frame_rate =
    config->get_value< double >( "frame_rate" );
  unsigned max_frame_count =
    config->get_value< unsigned >( "max_frame_count" );
  bool use_labels =
    config->get_value< bool >( "use_labels" );
  double downsample =
    config->get_value< double >( "downsample" );
  double targetted_downsample =
    config->get_value< double >( "targetted_downsample" );
  std::string targetted_downsample_string =
    config->get_value< std::string >( "targetted_downsample_string" );
  std::string secondary_frame_labels_str =
    config->get_value< std::string >( "secondary_frame_labels" );
  unsigned ignore_secondary_burst_count =
    config->get_value< unsigned >( "ignore_secondary_burst_count" );
  unsigned secondary_downsample =
    config->get_value< unsigned >( "secondary_downsample" );
  double threshold =
    config->get_value< double >( "threshold" );
  bool check_override =
    config->get_value< bool >( "check_override" );
  bool convert_to_full_frame =
    config->get_value< bool >( "convert_to_full_frame" );
  std::string data_warning_file =
    config->get_value< std::string >( "data_warning_file" );

  if( convert_to_full_frame && !kwiver::vital::algo::image_io::
        check_nested_algo_configuration( "image_reader", config ) )
  {
    std::cout << "Invalid image reader type specified" << std::endl;
    return EXIT_FAILURE;
  }

  if( targetted_downsample > 0 && targetted_downsample_string.empty() )
  {
    std::cout << "Target downsample string must be set to use target sampling" << std::endl;
    return EXIT_FAILURE;
  }

  if( !g_params.opt_threshold.empty() )
  {
    threshold = atof( g_params.opt_threshold.c_str() );
    std::cout << "Using command line provided threshold: " << threshold << std::endl;
  }

  if( !g_params.opt_pipeline_file.empty() )
  {
    pipeline_file = g_params.opt_pipeline_file;
  }

  if( !augmented_cache.empty() &&
      !pipeline_file.empty() &&
      create_folder( augmented_cache ) )
  {
    regenerate_cache = true;
  }

  std::unique_ptr< std::ofstream > data_warning_writer;
  std::vector< std::string > mentioned_warnings;

  if( !data_warning_file.empty() )
  {
    data_warning_writer.reset( new std::ofstream( data_warning_file.c_str() ) );
  }

  if( !g_params.opt_frame_rate.empty() )
  {
    frame_rate = std::stod( g_params.opt_frame_rate );
  }

  if( !g_params.opt_max_frame_count.empty() )
  {
    max_frame_count = std::stoi( g_params.opt_max_frame_count );
  }

  std::vector< std::string > image_exts, video_exts, groundtruth_exts;
  std::unordered_set< std::string > secondary_frame_labels;
  bool one_file_per_image;

  if( groundtruth_style == "one_per_file" )
  {
    one_file_per_image = true;
  }
  else if( groundtruth_style == "one_per_folder" )
  {
    one_file_per_image = false;
  }
  else
  {
    std::cerr << "Invalid groundtruth style: " << groundtruth_style << std::endl;
    return EXIT_FAILURE;
  }

  if( percent_validation < 0.0 || percent_validation > 1.0 )
  {
    std::cerr << "Percent validation must be [0.0,1.0]" << std::endl;
    return EXIT_FAILURE;
  }

  string_to_vector( image_exts_str, image_exts, "\n\t\v,; " );
  string_to_vector( video_exts_str, video_exts, "\n\t\v,; " );
  string_to_vector( groundtruth_exts_str, groundtruth_exts, "\n\t\v,; " );

  string_to_set( secondary_frame_labels_str, secondary_frame_labels, "\n\t\v,;" );

  // Load labels.txt file
  std::string label_fn;

  if( !g_params.opt_label_file.empty() )
  {
    label_fn = g_params.opt_label_file;
  }
  else if( !g_params.opt_input_dir.empty() )
  {
    label_fn = append_path( g_params.opt_input_dir, "labels.txt" );
  }

  kwiver::vital::category_hierarchy_sptr model_labels;
  bool detection_without_label = false;

  if( !does_file_exist( label_fn ) && g_params.opt_out_config.empty() )
  {
    std::cout << "Label file (labels.txt) does not exist in input folder" << std::endl;
    std::cout << std::endl << "Would you like to train over all category labels? (y/n) ";

    if( !g_params.opt_no_query )
    {
      std::string response;
      std::cin >> response;

      if( response != "y" && response != "Y" && response != "yes" && response != "Yes" )
      {
        std::cout << std::endl << "Exiting training due to no labels.txt" << std::endl;
        return EXIT_FAILURE;
      }
    }
  }
  else if( g_params.opt_out_config.empty() )
  {
    try
    {
      model_labels.reset( new kwiver::vital::category_hierarchy( label_fn ) );
    }
    catch( const std::exception& e )
    {
      std::cerr << "Error reading labels.txt: " << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Image reader and width/height only required for certain operations
  if( convert_to_full_frame )
  {
    kwiver::vital::algo::image_io::set_nested_algo_configuration
      ( "image_reader", config, image_reader );
    kwiver::vital::algo::image_io::get_nested_algo_configuration
      ( "image_reader", config, image_reader );
  }

  unsigned image_width = 0, image_height = 0;
  bool variable_resolution_sequences = false;

  // Data regardless of source
  std::vector< std::string > all_data;  // List of folders, image lists, or videos
  std::vector< std::string > all_truth; // Corresponding list of groundtruth files
  int validation_pivot = -1;            // Validation index start, if manually set
  bool auto_detect_truth = false;       // Auto-detect truth if not manually specified

  // Option 1: a typical training data directory is input
  if( !g_params.opt_input_dir.empty() )
  {
    std::string input_dir = g_params.opt_input_dir;

    if( !does_folder_exist( input_dir ) && does_folder_exist( input_dir + ".lnk" ) )
    {
      input_dir = filesystem::canonical(
        filesystem::path( input_dir + ".lnk" ) ).string();
    }

    if( !does_folder_exist( input_dir ) && g_params.opt_out_config.empty() )
    {
      std::cerr << "Input directory does not exist, exiting." << std::endl;
      return EXIT_FAILURE;
    }

    // Load train.txt, if available
    const std::string train_fn = append_path( input_dir, "train.txt" );

    if( does_file_exist( train_fn ) && !file_to_vector( train_fn, all_data ) )
    {
      std::cerr << "Unable to open " << train_fn << std::endl;
      return EXIT_FAILURE;
    }

    // Special use case for multiple overlapping streams (stereo, eo/ir fusion, etc..)
    const std::string train1_fn = append_path( input_dir, "train1.txt" );
    const std::string train2_fn = append_path( input_dir, "train2.txt" );

    if( does_file_exist( train1_fn ) )
    {
      if( does_file_exist( train_fn ) )
      {
        std::cerr << "Folder cannot contain both train.txt and train1.txt" << std::endl;
        return EXIT_FAILURE;
      }

      if( !file_to_vector( train1_fn, all_data ) )
      {
        std::cerr << "Unable to open " << label_fn << std::endl;
        return EXIT_FAILURE;
      }
    }

    // Note: Currently no option to use 2nd input stream in pipelines, just validate it
    std::vector< std::string > train2_files;
    if( does_file_exist( train2_fn ) && !file_to_vector( train2_fn, train2_files ) )
    {
      std::cerr << "Unable to open " << train2_fn << std::endl;
      return EXIT_FAILURE;
    }

    // Load validation.txt, if available
    std::string validation_fn = append_path( input_dir, "validation.txt" );

    if( does_file_exist( validation_fn ) )
    {
      validation_pivot = all_data.size();

      if( !file_to_vector( validation_fn, all_data, false ) )
      {
        std::cerr << "Unable to open " << validation_fn << std::endl;
        return EXIT_FAILURE;
      }
    }

    // First case, custom .txts are provided, second for just a directory
    if( !all_data.empty() )
    {
      // If validation set specified, confirm there is some training data
      if( !validation_pivot )
      {
        std::cerr << "If validation.txt is specified, so must a train.txt" << std::endl;
        return EXIT_FAILURE;
      }

      // Append path to all train and validation files, test to see if they all exist
      bool absolute_paths = false;
      std::string to_test = all_data[0];
      std::string full_path = append_path( g_params.opt_input_dir, to_test );

      if( !does_file_exist( full_path ) && does_file_exist( to_test ) )
      {
        absolute_paths = true;
        std::cout << "Using absolute paths in train.txt and validation.txt" << std::endl;
      }

      for( unsigned i = 0; i < all_data.size(); i++ )
      {
        if( !absolute_paths )
        {
          all_data[i] = append_path( g_params.opt_input_dir, all_data[i] );
        }

        if( !does_file_exist( all_data[i] ) )
        {
          std::cerr << "Could not find train file: " << all_data[i] << std::endl;
        }
      }
    }
    else
    {
      std::cout << "Automatically identifying train files from input directory" << std::endl;

      // Identify all sub-directories containing data
      std::vector< std::string > subfolders, videos;
      list_all_subfolders( g_params.opt_input_dir, subfolders );
      list_files_in_folder( g_params.opt_input_dir, videos, false, video_exts );

      all_data.insert( all_data.end(), subfolders.begin(), subfolders.end() );
      all_data.insert( all_data.end(), videos.begin(), videos.end() );

      if( all_data.empty() )
      {
        std::cout << "Error: training folder contains no sub-folders" << std::endl;
        return EXIT_FAILURE;
      }
    }

    auto_detect_truth = true;
  }
  else if( !g_params.opt_input_list.empty() )
  {
    if( !does_file_exist( g_params.opt_input_list ) ||
        !load_file_list( g_params.opt_input_list, all_data ) )
    {
      std::cout << "Unable to load: " << g_params.opt_input_list << std::endl;
      return EXIT_FAILURE;
    }

    while( !all_data.empty() && all_data.back().empty() )
    {
      all_data.pop_back();
    }

    if( all_data.empty() )
    {
      std::cout << "Input training data list contains no entries" << std::endl;
      return EXIT_FAILURE;
    }

    auto_detect_truth = g_params.opt_input_truth.empty();

    if( !auto_detect_truth )
    {
      // Check if input_truth is a single file (CSV) or a list file
      if( does_file_exist( g_params.opt_input_truth ) )
      {
        // Check if it's a groundtruth file directly (e.g., .csv) or a list file
        bool is_truth_file = ends_with_extension( g_params.opt_input_truth, groundtruth_exts );

        if( is_truth_file )
        {
          // Single truth file for all images - replicate it for each data entry
          all_truth.resize( all_data.size(), g_params.opt_input_truth );
        }
        else
        {
          // It's a list file containing paths to truth files
          if( !load_file_list( g_params.opt_input_truth, all_truth ) )
          {
            std::cout << "Unable to load: " << g_params.opt_input_truth << std::endl;
            return EXIT_FAILURE;
          }

          while( all_truth.size() > all_data.size() && all_truth.back().empty() )
          {
            all_truth.pop_back();
          }

          if( all_data.size() != all_truth.size() )
          {
            std::cout << "Training data and truth list lengths do not match" << std::endl;
            return EXIT_FAILURE;
          }
        }
      }
      else
      {
        std::cout << "Unable to find: " << g_params.opt_input_truth << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  // Load optional manual validation folder
  if( !g_params.opt_validation_dir.empty() )
  {
    std::vector< std::string > subfolders, videos;

    if( validation_pivot < 0 )
    {
      validation_pivot = all_data.size();
    }

    if( !does_folder_exist( g_params.opt_validation_dir ) )
    {
      std::cerr << "Unable to open " << g_params.opt_validation_dir << std::endl;
      return EXIT_FAILURE;
    }

    list_all_subfolders( g_params.opt_validation_dir, subfolders );
    list_files_in_folder( g_params.opt_validation_dir, videos, false, video_exts );

    all_data.insert( all_data.end(), subfolders.begin(), subfolders.end() );
    all_data.insert( all_data.end(), videos.begin(), videos.end() );
  }

  // Load groundtruth for all image files in all folders using reader class
  std::vector< std::string > train_image_fn;
  std::vector< kwiver::vital::detected_object_set_sptr > train_gt;
  std::vector< std::string > validation_image_fn;
  std::vector< kwiver::vital::detected_object_set_sptr > validation_gt;

  // Retain class counts for error checking
  std::map< std::string, int > label_counts;

  for( unsigned i = 0; i < all_data.size(); i++ )
  {
    // Get next data entry to process
    std::string data_item = all_data[i];
    std::cout << "Processing " << data_item << std::endl;

    // If train/validation partition divide already set, updated from
    // data sequence to image id level.
    if( validation_pivot == static_cast< int >( i ) )
    {
      validation_pivot = train_image_fn.size();
    }

    // Identify all truth files for this entry
    std::vector< std::string > image_files, video_files, gt_files;

    bool is_video = ends_with_extension( data_item, video_exts );

    if( is_video && auto_detect_truth )
    {
      std::string video_truth = replace_ext_with( data_item, groundtruth_exts[0] );

      if( !does_file_exist( video_truth ) )
      {
        std::string error_msg = "Error: cannot find " + video_truth;
        video_truth = add_ext_unto( data_item, groundtruth_exts[0] );

        if( !does_file_exist( video_truth ) )
        {
          std::cout << error_msg << std::endl;
          return EXIT_FAILURE;
        }
      }

      gt_files.resize( 1, video_truth );
    }
    else if( !is_video && auto_detect_truth )
    {
      list_files_in_folder( data_item, gt_files, false, groundtruth_exts );
      std::sort( gt_files.begin(), gt_files.end() );

      if( gt_files.empty() )
      {
        std::string truth = add_ext_unto( data_item, groundtruth_exts[0] );

        if( does_file_exist( truth ) )
        {
          gt_files.push_back( truth );
        }
      }

      if( one_file_per_image && ( image_files.size() != gt_files.size() ) )
      {
        std::cout << "Error: item " << data_item << " contains unequal truth and "
                  << "image file counts" << std::endl << " - Consider turning on "
                  << "the one_per_folder groundtruth style" << std::endl;
        return EXIT_FAILURE;
      }
      else if( gt_files.empty() )
      {
        std::cout << "Error reading item " << data_item << ", no groundtruth." << std::endl;
        return EXIT_FAILURE;
      }
    }
    else
    {
      gt_files.resize( 1, all_truth[i] );
    }

    // Either the input is a video file directly, a single image file, or a directory
    // containing images or video. In case of the latter, autodetect presence of images or video.
    bool is_image = ends_with_extension( data_item, image_exts );

    if( is_image )
    {
      // Single image file provided directly
      image_files.push_back( data_item );
    }
    else
    {
      list_files_in_folder( data_item, video_files, true, video_exts );

      if( !is_video )
      {
        list_files_in_folder( data_item, image_files, true, image_exts );

        if( video_files.size() == 1 && image_files.size() < 2 )
        {
          image_files.clear();
          is_video = true;
          data_item = video_files[0];
        }
      }
    }

    if( is_video )
    {
      double file_frame_rate = get_file_frame_rate( gt_files[0] );

      if( video_extractor.find( "_only" ) != std::string::npos )
      {
        std::cout << "Detection and track only frame extractors not yet supported" << std::endl;
        return EXIT_FAILURE;
      }

      image_files = extract_video_frames( data_item, video_extractor,
        ( file_frame_rate > 0 ? file_frame_rate : frame_rate ),
        augmented_cache, !regenerate_cache, max_frame_count );

      if( max_frame_count > 0 )
      {
        break;
      }
    }

    std::sort( image_files.begin(), image_files.end() );

    // Load groundtruth file for this entry
    kwiver::vital::algo::detected_object_set_input_sptr gt_reader;

    if( !one_file_per_image )
    {
      if( gt_files.size() != 1 )
      {
        std::cout << "Error: item " << data_item
                  << " must contain only 1 groundtruth file" << std::endl;
        return EXIT_FAILURE;
      }

      kwiver::vital::algo::detected_object_set_input::set_nested_algo_configuration
        ( "groundtruth_reader", config, gt_reader );
      kwiver::vital::algo::detected_object_set_input::get_nested_algo_configuration
        ( "groundtruth_reader", config, gt_reader );

      std::cout << "Opening groundtruth file " << gt_files[0] << std::endl;

      gt_reader->open( gt_files[0] );
    }

    // Perform any augmentation for this entry, if enabled
    pipeline_t augmentation_pipe = load_embedded_pipeline( pipeline_file );
    std::string last_subdir;

    if( !augmented_cache.empty() && !pipeline_file.empty() )
    {
      std::vector< std::string > cache_path, split_folder;
      kwiversys::SystemTools::SplitPath( data_item, split_folder );
      last_subdir = ( split_folder.empty() ? data_item : split_folder.back() );

      cache_path.push_back( "" );
      cache_path.push_back( augmented_cache );
      cache_path.push_back( last_subdir );

      create_folder( kwiversys::SystemTools::JoinPath( cache_path ) );
    }

    // Read all images and detections in sequence
    if( image_files.size() == 0 )
    {
      std::cout << "Error: entry " << data_item << " contains no image files." << std::endl;
    }

    for( unsigned i = 0; i < image_files.size(); ++i )
    {
      const std::string& image_file = image_files[i];

      bool use_image = true;
      std::string filtered_image_file;

      if( augmentation_pipe )
      {
        filtered_image_file = get_augmented_filename( image_file, last_subdir,
          augmented_cache, augmented_ext_override );

        if( regenerate_cache )
        {
          if( !run_pipeline_on_image( augmentation_pipe, pipeline_file,
                image_file, filtered_image_file ) )
          {
            use_image = false;
          }
        }
        else
        {
          use_image = filesystem::exists( filtered_image_file );
        }
      }
      else
      {
        filtered_image_file = image_file;
      }

      // Read groundtruth for image
      kwiver::vital::detected_object_set_sptr frame_dets =
        std::make_shared< kwiver::vital::detected_object_set>();

      if( one_file_per_image )
      {
        gt_reader.reset();

        kwiver::vital::algo::detected_object_set_input::set_nested_algo_configuration
          ( "groundtruth_reader", config, gt_reader );
        kwiver::vital::algo::detected_object_set_input::get_nested_algo_configuration
          ( "groundtruth_reader", config, gt_reader );

        gt_reader->open( gt_files[i] );

        std::string read_fn = get_filename_no_path( image_file );

        gt_reader->read_set( frame_dets, read_fn );
        gt_reader->close();
      }
      else
      {
        std::string read_fn = get_filename_no_path( image_file );

        try
        {
          gt_reader->read_set( frame_dets, read_fn );
        }
        catch( const std::exception& e )
        {
          std::cerr << "Received exception: " << e.what() << std::endl
                    << "Unable to load groundtruth file: " << read_fn << std::endl;
          return EXIT_FAILURE;
        }
      }

      correct_manual_annotations( frame_dets );

      if( convert_to_full_frame )
      {
        if( i < 4 || variable_resolution_sequences )
        {
          auto image = image_reader->load( image_file );

          unsigned new_width = image->width();
          unsigned new_height = image->height();

          if( i > 0 && ( new_width != image_width || new_height != image_height ) )
          {
            variable_resolution_sequences = true;
          }

          image_width = new_width;
          image_height = new_height;
        }

        frame_dets = adjust_to_full_frame( frame_dets, image_width, image_height );
      }

      // Apply threshold to frame detections
      if( use_image )
      {
        std::cout << "Read " << frame_dets->size()
                  << " detections for "
                  << image_file << std::endl;

        kwiver::vital::detected_object_set_sptr filtered_dets =
          std::make_shared< kwiver::vital::detected_object_set>();

        for( auto det : *frame_dets )
        {
          bool add_detection = false;
          auto class_scores = det->type();

          if( class_scores )
          {
            for( auto gt_class : class_scores->class_names() )
            {
              if( !model_labels || model_labels->has_class_name( gt_class ) )
              {
                if( class_scores->score( gt_class ) >= threshold )
                {
                  if( model_labels )
                  {
                    gt_class = model_labels->get_class_name( gt_class );
                  }

                  label_counts[ gt_class ]++;
                  add_detection = true;
                }
              }
              else
              {
                class_scores->delete_score( gt_class );

                if( data_warning_writer &&
                    std::find(
                      mentioned_warnings.begin(),
                      mentioned_warnings.end(),
                      gt_class ) == mentioned_warnings.end() )
                {
                  *data_warning_writer << "Observed class: "
                    << gt_class << " not in input labels.txt" << std::endl;

                  mentioned_warnings.push_back( gt_class );
                }
              }
            }
          }
          else if( !model_labels || model_labels->size() == 1 )
          {
            add_detection = true; // single class problem, doesn't need dot
            detection_without_label = true; // at least 1 detection lacks a label
          }

          if( add_detection )
          {
            filtered_dets->add( det );
          }
        }

        train_image_fn.push_back( filtered_image_file );
        train_gt.push_back( filtered_dets );
      }

      if( max_frame_count > 0 && train_image_fn.size() > max_frame_count )
      {
        break;
      }
    }

    if( augmentation_pipe )
    {
      augmentation_pipe->send_end_of_input();
      augmentation_pipe->wait();
    }

    if( !one_file_per_image )
    {
      gt_reader->close();
    }

    if( max_frame_count > 0 && train_image_fn.size() > max_frame_count )
    {
      break;
    }
  }

  if( validation_pivot > 0 )
  {
    validation_image_fn.insert( validation_image_fn.begin(),
      train_image_fn.begin() + validation_pivot, train_image_fn.end() );
    validation_gt.insert( validation_gt.begin(),
      train_gt.begin() + validation_pivot, train_gt.end() );

    train_image_fn.erase(
      train_image_fn.begin() + validation_pivot, train_image_fn.end() );
    train_gt.erase(
      train_gt.begin() + validation_pivot, train_gt.end() );
  }

  if( downsample > 0 )
  {
    downsample_data( train_image_fn, train_gt, downsample );
  }

  if( targetted_downsample > 0 )
  {
    downsample_data( train_image_fn, train_gt, targetted_downsample, targetted_downsample_string );
  }

  if( label_counts.empty() )
  {
    for( auto det_set : train_gt )
    {
      for( auto det : *det_set )
      {
        if( det->type() )
        {
          std::string gt_class;
          det->type()->get_most_likely( gt_class );

          if( !model_labels || model_labels->has_class_name( gt_class ) )
          {
            if( model_labels )
            {
              gt_class = model_labels->get_class_name( gt_class );
            }

            label_counts[ gt_class ]++;
          }
          else if( data_warning_writer &&
                  std::find(
                    mentioned_warnings.begin(),
                    mentioned_warnings.end(),
                    gt_class ) == mentioned_warnings.end() )
          {
            *data_warning_writer << "Observed class: "
               << gt_class << " not in input labels.txt" << std::endl;

            mentioned_warnings.push_back( gt_class );
          }
        }
      }
    }
  }

  if( label_counts.empty() ) // groundtruth has no classification labels
  {
    // Only 1 class, is okay but inject the classification into the groundtruth
    if( !model_labels || model_labels->size() == 1 )
    {
      std::string label = model_labels ? model_labels->all_class_names()[0] : "object";

      for( auto det_set : train_gt )
      {
        for( auto det : *det_set )
        {
          det->set_type(
            kwiver::vital::detected_object_type_sptr(
              new kwiver::vital::detected_object_type( label, 1.0 ) ) );

          label_counts[ label ]++;
        }
      }
      for( auto det_set : validation_gt )
      {
        for( auto det : *det_set )
        {
          det->set_type(
            kwiver::vital::detected_object_type_sptr(
              new kwiver::vital::detected_object_type( label, 1.0 ) ) );
        }
      }
    }
    else // Not okay
    {
      std::cout << "Error: input labels.txt contains multiple classes, but supplied "
                << "truth files do not contain the training classes of interest, or "
                << "there was an error reading them from the input annotations."
                << std::endl;

      return EXIT_FAILURE;
    }
  }
  else if( !check_override && model_labels )
  {
    for( auto cls : model_labels->all_class_names() )
    {
      if( label_counts[ cls ] == 0 )
      {
        std::cout << "Error: no entries in groundtruth of class " << cls << std::endl
                  << std::endl
                  << "Optionally set \"check_override\" parameter to ignore this check."
                  << std::endl;

        return EXIT_FAILURE;
      }
    }
  }
  else if( detection_without_label )
  {
    std::cout << "Warning: one or more annotations contain no class label specified"
              << std::endl
              << "Consider checking your groundtruth for consisitency"
              << std::endl;
  }

  if( !model_labels )
  {
    model_labels.reset( new kwiver::vital::category_hierarchy() );

    int id = 0;

    for( auto label : label_counts )
    {
      model_labels->add_class( label.first, "", id++ );
    }
  }

  // Use GT frames only if enabled
  if( g_params.opt_gt_only )
  {
    std::vector< std::string > adj_train_image_fn;
    std::vector< kwiver::vital::detected_object_set_sptr > adj_train_gt;

    for( unsigned i = 0; i < train_image_fn.size(); ++i )
    {
      if( !train_gt[i]->empty() )
      {
        adj_train_image_fn.push_back( train_image_fn[i] );
        adj_train_gt.push_back( train_gt[i] );
      }
    }

    train_image_fn = adj_train_image_fn;
    train_gt = adj_train_gt;
  }

  // Generate a validation set automatically if enabled
  bool invalid_train_set = false, invalid_validation_set = false, found_any = false;

  if( percent_validation > 0.0 && validation_image_fn.empty() )
  {
    unsigned total_images = train_image_fn.size();

    unsigned total_segment = static_cast< unsigned >( validation_burst_frame_count / percent_validation );
    unsigned train_segment = total_segment - validation_burst_frame_count;

    if( total_images < total_segment )
    {
      total_segment = total_images;
      train_segment = total_images - static_cast< unsigned >( percent_validation * total_images );

      if( total_segment > 1 && train_segment == total_segment )
      {
        train_segment = total_segment - 1;
      }
    }

    bool found_first = false, found_second = false, initial_override = false;
    std::vector< std::string > adj_train_image_fn;
    std::vector< kwiver::vital::detected_object_set_sptr > adj_train_gt;

    for( unsigned i = 0; i < train_image_fn.size(); ++i )
    {
      // First 2 conditionals are hack to ensure at least 1 truth frame.
      if( !found_first && !train_gt[i]->empty() )
      {
        validation_image_fn.push_back( train_image_fn[i] );
        validation_gt.push_back( train_gt[i] );
        found_first = true;
        found_any = true;
      }
      else if( !found_second && !train_gt[i]->empty() )
      {
        adj_train_image_fn.push_back( train_image_fn[i] );
        adj_train_gt.push_back( train_gt[i] );
        found_second = true;
        initial_override = true;
      }
      else if( initial_override || i % total_segment < train_segment )
      {
        if( initial_override && i % train_segment == 0 )
        {
          initial_override = false;
        }
        adj_train_image_fn.push_back( train_image_fn[i] );
        adj_train_gt.push_back( train_gt[i] );
      }
      else
      {
        validation_image_fn.push_back( train_image_fn[i] );
        validation_gt.push_back( train_gt[i] );
      }
    }

    invalid_validation_set = !found_first;
    invalid_train_set = !found_second;

    train_image_fn = adj_train_image_fn;
    train_gt = adj_train_gt;
  }

  // Backup case for small datasets
  if( percent_validation > 0.0 )
  {
    invalid_validation_set = is_detection_set_empty( validation_gt );

    if( !train_image_fn.empty() &&
       ( validation_image_fn.empty() || invalid_validation_set ) )
    {
      for( unsigned i = 0; i < train_image_fn.size() - 1; i++ )
      {
        validation_image_fn.push_back( train_image_fn.back() );
        validation_gt.push_back( train_gt.back() );

        train_image_fn.pop_back();
        train_gt.pop_back();

        if( validation_gt.back() && !validation_gt.back()->empty() )
        {
          invalid_validation_set = false;
          break;
        }
      }
    }

    invalid_train_set = is_detection_set_empty( train_gt );
  }

  // Final validation checks
  if( !found_any )
  {
    for( const auto& dets : train_gt )
    {
      if( !dets->empty() )
      {
        found_any = true;
        break;
      }
    }

    if( !found_any )
    {
      std::cout << "Error: training set contains no truth detections to use. "
        "Check to make sure you have appropriately formatted inputs." << std::endl;
      return EXIT_FAILURE;
    }
  }

  if( invalid_train_set || invalid_validation_set )
  {
    std::cout << "Error: Either not enough input diversity to train model, "
      "or improperly formatted input supplied." << std::endl;
    return EXIT_FAILURE;
  }

  // Adjust labels 
  if( use_labels )
  {
    std::vector< bool > fg_mask = adjust_labels( train_gt,
      model_labels, secondary_frame_labels );

    adjust_labels( train_image_fn, train_gt, fg_mask,
      secondary_downsample, ignore_secondary_burst_count );

    adjust_labels( validation_gt, model_labels, secondary_frame_labels );
  }

  // Run training algorithm(s) - loop through all configs/detectors for multi-model training
  for( unsigned model_idx = 0; model_idx < model_count; ++model_idx )
  {
    if( multi_model_training )
    {
      std::cout << std::endl << "========================================" << std::endl;
      std::cout << "Training model " << ( model_idx + 1 ) << " of " << model_count << std::endl;
      std::cout << "========================================" << std::endl;

      // Reconfigure for this model
      kwiver::vital::config_block_sptr model_config = default_config();

      if( !training_configs.empty() )
      {
        std::string current_config = training_configs[ model_idx ];
        std::cout << "Using config: " << current_config << std::endl;

        try
        {
          model_config->merge_config( kwiver::vital::read_config_file( current_config ) );
        }
        catch( const std::exception& e )
        {
          std::cerr << "Received exception: " << e.what() << std::endl
                    << "Unable to load configuration file: "
                    << current_config << std::endl;
          continue;
        }
      }
      else
      {
        std::string current_detector = training_detectors[ model_idx ];
        std::cout << "Using detector type: " << current_detector << std::endl;
        model_config->set_value( "detector_trainer:type", current_detector );
      }

      // Apply command line settings override
      if( !g_params.opt_settings.empty() )
      {
        const std::string& setting = g_params.opt_settings;
        size_t const split_pos = setting.find( "=" );

        if( split_pos != std::string::npos )
        {
          kwiver::vital::config_block_key_t setting_key =
            setting.substr( 0, split_pos );
          kwiver::vital::config_block_value_t setting_value =
            setting.substr( split_pos + 1 );
          model_config->set_value( setting_key, setting_value );
        }
      }

      // Reinitialize detector trainer for this model
      detector_trainer.reset();
      kwiver::vital::algo::train_detector::set_nested_algo_configuration
        ( "detector_trainer", model_config, detector_trainer );
      kwiver::vital::algo::train_detector::get_nested_algo_configuration
        ( "detector_trainer", model_config, detector_trainer );

      if( !kwiver::vital::algo::train_detector::
            check_nested_algo_configuration( "detector_trainer", model_config ) )
      {
        std::cout << "Configuration not valid for model " << ( model_idx + 1 ) << std::endl;
        continue;
      }
    }

    std::cout << "Beginning Training Process" << std::endl;
    std::string error;

    try
    {
      detector_trainer->add_data_from_disk( model_labels,
        train_image_fn, train_gt, validation_image_fn, validation_gt );

      detector_trainer->update_model();
    }
    catch( const std::exception& e )
    {
      error = e.what();
    }
    catch( const std::string& str )
    {
      error = str;
    }
    catch( ... )
    {
      error = "unknown fault";
    }

    if( !error.empty() )
    {
      if( error.find( "interupt_handler" ) != std::string::npos ||
          error.find( "KeyboardInterrupt" ) != std::string::npos )
      {
        std::cout << "Finished spooling down run after interrupt" << std::endl << std::endl;
        break; // Exit loop on interrupt
      }
      else
      {
        std::cout << "Received exception: " << error << std::endl;
        std::cout << std::endl;
        if( multi_model_training )
        {
          std::cout << "Continuing to next model..." << std::endl << std::endl;
        }
        else
        {
          std::cout << "Shutting down" << std::endl << std::endl;
        }
      }
    }
    else if( multi_model_training )
    {
      std::cout << "Model " << ( model_idx + 1 ) << " training completed successfully"
                << std::endl;
    }
  }

  if( multi_model_training )
  {
    std::cout << std::endl << "========================================" << std::endl;
    std::cout << "Multi-model training complete" << std::endl;
    std::cout << "========================================" << std::endl;
  }

  // Tracker training section (optional, runs after detector training if enabled)
  if( train_trackers )
  {
    std::cout << std::endl << "========================================" << std::endl;
    std::cout << "Beginning Tracker Training" << std::endl;
    std::cout << "========================================" << std::endl;

    // Read track groundtruth directly using track_reader
    // Note: Tracker training uses the same groundtruth files but reads them as tracks
    // (with track IDs preserved) rather than as per-frame detections
    std::vector< kwiver::vital::object_track_set_sptr > train_tracks;
    std::vector< kwiver::vital::object_track_set_sptr > validation_tracks;

    // Configure track reader
    kwiver::vital::algo::read_object_track_set::set_nested_algo_configuration
      ( "track_reader", config, track_reader );

    if( track_reader )
    {
      // Re-read groundtruth files as tracks for training data
      std::cout << "Reading track groundtruth for training..." << std::endl;

      for( unsigned i = 0; i < all_data.size(); i++ )
      {
        std::string data_item = all_data[i];
        bool is_validation = ( validation_pivot >= 0 &&
                               static_cast< int >( i ) >= validation_pivot );

        // Find groundtruth file for this data entry
        std::vector< std::string > gt_files;
        bool is_video = ends_with_extension( data_item, video_exts );

        if( is_video && auto_detect_truth )
        {
          std::string video_truth = replace_ext_with( data_item, groundtruth_exts[0] );
          if( !does_file_exist( video_truth ) )
          {
            video_truth = add_ext_unto( data_item, groundtruth_exts[0] );
          }
          if( does_file_exist( video_truth ) )
          {
            gt_files.push_back( video_truth );
          }
        }
        else if( !is_video && auto_detect_truth )
        {
          list_files_in_folder( data_item, gt_files, false, groundtruth_exts );
          std::sort( gt_files.begin(), gt_files.end() );

          if( gt_files.empty() )
          {
            std::string truth = add_ext_unto( data_item, groundtruth_exts[0] );
            if( does_file_exist( truth ) )
            {
              gt_files.push_back( truth );
            }
          }
        }
        else if( i < all_truth.size() )
        {
          gt_files.push_back( all_truth[i] );
        }

        // Read tracks from each groundtruth file
        for( const auto& gt_file : gt_files )
        {
          try
          {
            track_reader->open( gt_file );

            kwiver::vital::object_track_set_sptr tracks;
            if( track_reader->read_set( tracks ) && tracks )
            {
              std::cout << "Read " << tracks->size() << " tracks from "
                        << gt_file << std::endl;

              if( is_validation )
              {
                validation_tracks.push_back( tracks );
              }
              else
              {
                train_tracks.push_back( tracks );
              }
            }

            track_reader->close();
          }
          catch( const std::exception& e )
          {
            std::cerr << "Warning: Could not read tracks from " << gt_file
                      << ": " << e.what() << std::endl;
          }
        }
      }

      std::cout << "Loaded " << train_tracks.size() << " training track sets, "
                << validation_tracks.size() << " validation track sets" << std::endl;
    }
    else
    {
      std::cout << "Warning: No track reader configured, tracker training may fail"
                << std::endl;
    }

    // Train each specified tracker
    for( unsigned tracker_idx = 0; tracker_idx < training_trackers.size(); ++tracker_idx )
    {
      std::string current_tracker = training_trackers[ tracker_idx ];
      std::cout << std::endl << "Training tracker: " << current_tracker << std::endl;

      // Configure tracker trainer
      kwiver::vital::config_block_sptr tracker_config = default_config();
      tracker_config->set_value( "tracker_trainer:type", current_tracker );

      // Apply command line settings override
      if( !g_params.opt_settings.empty() )
      {
        const std::string& setting = g_params.opt_settings;
        size_t const split_pos = setting.find( "=" );

        if( split_pos != std::string::npos )
        {
          kwiver::vital::config_block_key_t setting_key =
            setting.substr( 0, split_pos );
          kwiver::vital::config_block_value_t setting_value =
            setting.substr( split_pos + 1 );
          tracker_config->set_value( setting_key, setting_value );
        }
      }

      kwiver::vital::algo::train_tracker::set_nested_algo_configuration
        ( "tracker_trainer", tracker_config, tracker_trainer );
      kwiver::vital::algo::train_tracker::get_nested_algo_configuration
        ( "tracker_trainer", tracker_config, tracker_trainer );

      if( !kwiver::vital::algo::train_tracker::
            check_nested_algo_configuration( "tracker_trainer", tracker_config ) )
      {
        std::cout << "Configuration not valid for tracker: " << current_tracker << std::endl;
        continue;
      }

      std::string error;

      try
      {
        tracker_trainer->add_data_from_disk( model_labels,
          train_image_fn, train_tracks, validation_image_fn, validation_tracks );

        tracker_trainer->update_model();
      }
      catch( const std::exception& e )
      {
        error = e.what();
      }
      catch( const std::string& str )
      {
        error = str;
      }
      catch( ... )
      {
        error = "unknown fault";
      }

      if( !error.empty() )
      {
        if( error.find( "interupt_handler" ) != std::string::npos ||
            error.find( "KeyboardInterrupt" ) != std::string::npos )
        {
          std::cout << "Finished spooling down run after interrupt" << std::endl << std::endl;
          break;
        }
        else
        {
          std::cout << "Received exception: " << error << std::endl;
          std::cout << std::endl;
        }
      }
      else
      {
        std::cout << "Tracker training completed successfully" << std::endl;
      }
    }

    std::cout << std::endl << "========================================" << std::endl;
    std::cout << "Tracker training complete" << std::endl;
    std::cout << "========================================" << std::endl;
  }

  return EXIT_SUCCESS;
}
