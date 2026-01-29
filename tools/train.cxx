/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "train.h"

#include <kwiversys/SystemTools.hxx>

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
#include <vital/algo/image_object_detector.h>
#include <vital/algo/read_object_track_set.h>
#include <vital/algo/image_io.h>
#include <vital/types/image_container.h>
#include <vital/types/object_track_set.h>
#include <vital/logger/logger.h>

#include <sprokit/pipeline/process_exception.h>

#include <plugins/core/utilities_file.h>
#include <plugins/core/utilities_training.h>
#include <plugins/core/manipulate_pipelines.h>

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

namespace kv = kwiver::vital;

namespace viame {
namespace tools {

// =======================================================================================
// Assorted configuration related helper functions
static kv::config_block_sptr default_config()
{
  kv::config_block_sptr config
    = kv::config_block::empty_config( "detector_trainer_tool" );

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
  config->set_value( "output_directory", "",
    "Directory to store trained model files and generated pipelines. "
    "If empty and output_file is not set, files are written to the current directory." );
  config->set_value( "output_file", "",
    "If specified, create a zip file containing the model files and pipeline "
    "instead of writing to output_directory. Takes precedence over output_directory." );
  config->set_value( "pipeline_template", "",
    "Optional template file for generating output pipeline. Keywords in the "
    "template will be replaced with values from the trainer." );
  config->set_value( "output_pipeline_name", "detector.pipe",
    "Name for the generated output pipeline file." );

  kv::algo::detected_object_set_input::get_nested_algo_configuration
    ( "groundtruth_reader", config, kv::algo::detected_object_set_input_sptr() );
  kv::algo::image_io::get_nested_algo_configuration
    ( "image_reader", config, kv::algo::image_io_sptr() );
  kv::algo::train_detector::get_nested_algo_configuration
    ( "detector_trainer", config, kv::algo::train_detector_sptr() );
  kv::algo::train_tracker::get_nested_algo_configuration
    ( "tracker_trainer", config, kv::algo::train_tracker_sptr() );
  kv::algo::read_object_track_set::get_nested_algo_configuration
    ( "track_reader", config, kv::algo::read_object_track_set_sptr() );

  return config;
}

// =======================================================================================
// Validate that trainer output keys match the inference algorithm's config.
// Returns true if all keys are valid, false otherwise.
static bool validate_trainer_output_keys(
    const std::map< std::string, std::string >& output_map,
    const std::string& algorithm_type,
    bool is_detector )
{
  if( output_map.empty() || algorithm_type.empty() )
  {
    return true;
  }

  // Try to get the inference algorithm's default configuration
  kv::config_block_sptr algo_config;

  try
  {
    if( is_detector )
    {
      kv::algo::image_object_detector_sptr detector;

      kv::config_block_sptr temp_config = kv::config_block::empty_config();
      temp_config->set_value( "detector:type", algorithm_type );
      kv::algo::image_object_detector::set_nested_algo_configuration(
        "detector", temp_config, detector );

      if( detector )
      {
        algo_config = kv::config_block::empty_config();
        kv::algo::image_object_detector::get_nested_algo_configuration(
          "detector", algo_config, detector );
      }
    }
    else
    {
      // For trackers, validation is less strict since tracker configs vary more
      return true;
    }
  }
  catch( ... )
  {
    // If we can't instantiate the algorithm, skip validation
    std::cerr << "Warning: Could not validate output keys against algorithm '"
              << algorithm_type << "'" << std::endl;
    return true;
  }

  if( !algo_config )
  {
    return true;
  }

  // Check each non-file key against the algorithm config
  bool all_valid = true;
  for( const auto& pair : output_map )
  {
    const std::string& key = pair.first;
    const std::string& value = pair.second;

    // Skip if value is an existing file (it's a file copy, not a config key)
    if( !value.empty() && does_file_exist( value ) )
    {
      continue;
    }

    // 'eval' and 'type' are special keys that don't need to be in the inference config.
    // 'type' specifies the algorithm type in the pipeline template.
    if( key == "eval" || key == "type" )
    {
      continue;
    }

    // Skip nested algorithm config keys (e.g., "ocv_windowed:detector:netharn:deployed").
    // These can't be validated without knowing and instantiating the nested algorithm.
    std::string algo_prefix = algorithm_type + ":";
    std::string nested_prefix = algo_prefix + "detector:";
    if( key.find( nested_prefix ) == 0 )
    {
      continue;
    }

    // Check if the key exists in the algorithm's config
    // If the key already starts with the algorithm prefix (e.g., "ocv_windowed:mode"),
    // don't add it again. The trainer outputs prefixed keys for template replacement.
    std::string full_key;
    if( key.find( algo_prefix ) == 0 )
    {
      // Key already has algorithm prefix, just add "detector:"
      full_key = "detector:" + key;
    }
    else
    {
      // Key doesn't have algorithm prefix, add both
      full_key = "detector:" + algorithm_type + ":" + key;
    }

    if( !algo_config->has_value( full_key ) )
    {
      std::cerr << "Error: Trainer returned key '" << key
                << "' which is not a valid config key for algorithm '"
                << algorithm_type << "'" << std::endl;
      all_valid = false;
    }
  }

  return all_valid;
}

// =======================================================================================
// Process the output map returned by a trainer's update_model() method.
// - If the value is an existing file path, it's a file copy (key=output filename)
// - Otherwise, it's a template replacement (key becomes [-KEY-] in template)
// - If output_file is specified, creates a zip archive instead of writing to directory
static void process_trainer_output(
    const std::map< std::string, std::string >& output_map,
    const std::string& output_directory,
    const std::string& output_file,
    const std::string& pipeline_template,
    const std::string& output_pipeline_name,
    const std::string& algorithm_type = "",
    bool is_detector = true )
{
  if( output_map.empty() )
  {
    return;
  }

  // Validate output keys against the inference algorithm's config
  if( !algorithm_type.empty() )
  {
    if( !validate_trainer_output_keys( output_map, algorithm_type, is_detector ) )
    {
      std::cerr << "Error: Trainer output contains invalid keys for algorithm '"
                << algorithm_type << "'" << std::endl;
      return;
    }
  }

  // Separate template replacements from file copies based on whether value is a file
  std::map< std::string, std::string > template_replacements;
  std::map< std::string, std::string > file_copies;

  for( const auto& pair : output_map )
  {
    const std::string& key = pair.first;
    const std::string& value = pair.second;

    // If value is an existing file, treat as file copy
    if( !value.empty() && does_file_exist( value ) )
    {
      file_copies[ key ] = value;
    }
    else
    {
      // Strip group prefix (everything up to and including the last ':')
      // so that "netharn:deployed" becomes "deployed", producing [-DEPLOYED-]
      std::string param_name = key;
      std::size_t last_colon = key.rfind( ':' );
      if( last_colon != std::string::npos )
      {
        param_name = key.substr( last_colon + 1 );
      }

      // Build template key: convert lowercase param to [-PARAM-] format
      std::string template_key = "[-";
      for( char c : param_name )
      {
        if( c == '_' )
        {
          template_key += '-';
        }
        else
        {
          template_key += std::toupper( static_cast< unsigned char >( c ) );
        }
      }
      template_key += "-]";
      template_replacements[ template_key ] = value;
    }
  }

  // Build [-DETECTOR-IMPL-] replacement from full trainer output
  std::string impl = generate_detector_impl_replacement(
      output_map, pipeline_template );

  if( !impl.empty() )
  {
    template_replacements[ "[-DETECTOR-IMPL-]" ] = impl;
  }

  // If output_file is specified, create a zip archive
  if( !output_file.empty() )
  {
    std::map< std::string, std::string > zip_files;
    std::map< std::string, std::string > zip_string_contents;

    // Add all model files to be included in zip
    for( const auto& pair : file_copies )
    {
      const std::string& dest_filename = pair.first;
      const std::string& source_path = pair.second;
      zip_files[ dest_filename ] = source_path;
    }

    // Generate pipeline content if template is configured
    if( !pipeline_template.empty() && does_file_exist( pipeline_template ) )
    {
      std::string pipeline_content;
      if( replace_keywords_in_template_to_string(
            pipeline_template, template_replacements, pipeline_content ) )
      {
        zip_string_contents[ output_pipeline_name ] = pipeline_content;
      }
      else
      {
        std::cerr << "Warning: failed to generate pipeline from template" << std::endl;
      }
    }

    // Create the zip file
    if( create_zip_file( output_file, zip_files, zip_string_contents ) )
    {
      std::cout << "Created output zip file: " << output_file << std::endl;
      std::cout << "  - Contains " << zip_files.size() << " model file(s)" << std::endl;
      if( !zip_string_contents.empty() )
      {
        std::cout << "  - Contains generated pipeline: " << output_pipeline_name << std::endl;
      }
    }
    else
    {
      std::cerr << "Error: failed to create zip file: " << output_file << std::endl;
    }

    return;
  }

  // Otherwise, use output_directory (existing behavior)
  // Create output directory if needed
  if( !output_directory.empty() )
  {
    create_folder( output_directory );
  }

  // Copy model files to output directory
  for( const auto& pair : file_copies )
  {
    const std::string& dest_filename = pair.first;
    const std::string& source_path = pair.second;

    std::string dest_path = output_directory.empty() ?
      dest_filename : append_path( output_directory, dest_filename );

    if( copy_file( source_path, dest_path ) )
    {
      std::cout << "Copied model file: " << dest_filename << std::endl;
    }
    else
    {
      std::cerr << "Warning: failed to copy " << source_path
                << " to " << dest_path << std::endl;
    }
  }

  // Generate pipeline from template if configured
  if( !pipeline_template.empty() && does_file_exist( pipeline_template ) )
  {
    std::string output_pipeline = output_directory.empty() ?
      output_pipeline_name : append_path( output_directory, output_pipeline_name );

    if( replace_keywords_in_template_file(
          pipeline_template, output_pipeline, template_replacements ) )
    {
      std::cout << "Generated pipeline: " << output_pipeline << std::endl;
    }
    else
    {
      std::cerr << "Warning: failed to generate pipeline from template" << std::endl;
    }
  }
}

// =======================================================================================
train_applet
::train_applet()
{
}

// =======================================================================================
void
train_applet
::add_command_options()
{
  m_cmd_options->add_options()
    ( "h,help", "Display usage information",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "l,list", "Display list of all trainable algorithms",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "no-query", "Do not query the user for anything",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "no-adv-prints", "Do not print out any advanced chars",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "no-embedded-pipe", "Do not output embedded pipes",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "gt-frames-only", "Use frames with annotations only",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "c,config", "Input configuration file(s) with parameters",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "i,input", "Input directory containing groundtruth",
      ::cxxopts::value< std::string >()->default_value( "" ), "dir" )
    ( "input-list", "Input list with data for training",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "input-truth", "Input list containing training truth",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "labels", "Input label file for train categories",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "v,validation", "Optional validation input directory",
      ::cxxopts::value< std::string >()->default_value( "" ), "dir" )
    ( "d,detector", "Type of detector(s) to train if no config",
      ::cxxopts::value< std::string >()->default_value( "" ), "type" )
    ( "tracker", "Type of tracker(s) to train (optional)",
      ::cxxopts::value< std::string >()->default_value( "" ), "type" )
    ( "o,output-config", "Output a sample configuration to file",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "s,setting", "Over-ride some setting in the config",
      ::cxxopts::value< std::string >()->default_value( "" ), "key=value" )
    ( "t,threshold", "Threshold override to apply over input",
      ::cxxopts::value< std::string >()->default_value( "" ), "value" )
    ( "p,pipeline", "Pipeline file",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "default-vfr", "Default video frame rate for extraction",
      ::cxxopts::value< std::string >()->default_value( "" ), "rate" )
    ( "max-frame-count", "Maximum frame count to use",
      ::cxxopts::value< std::string >()->default_value( "" ), "count" )
    ( "timeout", "Maximum time in seconds",
      ::cxxopts::value< std::string >()->default_value( "" ), "seconds" )
    ( "init-weights", "Optional input seed weights over-ride",
      ::cxxopts::value< std::string >()->default_value( "" ), "path" )
    ( "output-file", "Output zip file for model and pipeline (overrides output-dir)",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ;
}

// =======================================================================================
int
train_applet
::run()
{
  // Get logger
  kv::logger_handle_t logger = kv::get_logger( "viame.tools.train" );

  // Get command line arguments
  auto& cmd_args = command_args();

  // Print help
  if( cmd_args[ "help" ].as< bool >() )
  {
    std::cout << "Usage: viame train [options]\n"
              << "\nTrain one of several object detectors in the system.\n"
              << m_cmd_options->help() << std::endl;
    return EXIT_FAILURE;
  }

  // Extract options
  bool opt_list = cmd_args[ "list" ].as< bool >();
  bool opt_no_query = cmd_args[ "no-query" ].as< bool >();
  bool opt_no_adv_print = cmd_args[ "no-adv-prints" ].as< bool >();
  bool opt_no_emb_pipe = cmd_args[ "no-embedded-pipe" ].as< bool >();
  bool opt_gt_only = cmd_args[ "gt-frames-only" ].as< bool >();

  std::string opt_config = cmd_args[ "config" ].as< std::string >();
  std::string opt_input_dir = cmd_args[ "input" ].as< std::string >();
  std::string opt_input_list = cmd_args[ "input-list" ].as< std::string >();
  std::string opt_input_truth = cmd_args[ "input-truth" ].as< std::string >();
  std::string opt_label_file = cmd_args[ "labels" ].as< std::string >();
  std::string opt_validation_dir = cmd_args[ "validation" ].as< std::string >();
  std::string opt_detector = cmd_args[ "detector" ].as< std::string >();
  std::string opt_tracker = cmd_args[ "tracker" ].as< std::string >();
  std::string opt_out_config = cmd_args[ "output-config" ].as< std::string >();
  std::string opt_settings = cmd_args[ "setting" ].as< std::string >();
  std::string opt_threshold = cmd_args[ "threshold" ].as< std::string >();
  std::string opt_pipeline_file = cmd_args[ "pipeline" ].as< std::string >();
  std::string opt_frame_rate = cmd_args[ "default-vfr" ].as< std::string >();
  std::string opt_max_frame_count = cmd_args[ "max-frame-count" ].as< std::string >();
  std::string opt_timeout = cmd_args[ "timeout" ].as< std::string >();
  std::string opt_init_weights = cmd_args[ "init-weights" ].as< std::string >();
  std::string opt_output_file = cmd_args[ "output-file" ].as< std::string >();

  // List option
  if( opt_list )
  {
    kv::plugin_manager& vpm = kv::plugin_manager::instance();
    vpm.load_all_plugins();

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
      if( fact->get_attribute( kv::plugin_factory::PLUGIN_NAME, name ) )
      {
        std::cout << name << std::endl;
      }
    }
    return EXIT_FAILURE;
  }

  // Test for presence of conflicting options
  if( !opt_config.empty() && !opt_detector.empty() )
  {
    std::cerr << "Only one of --config and --detector allowed." << std::endl;
    return EXIT_FAILURE;
  }

  // Test for presence of required options (either detector or tracker training)
  if( opt_config.empty() && opt_detector.empty() &&
      opt_tracker.empty() )
  {
    std::cerr << "One of --config, --detector, or --tracker must be set." << std::endl;
    return EXIT_FAILURE;
  }

  // Parse comma-separated configs or detectors/trackers for multi-model training
  std::vector< std::string > training_configs;
  std::vector< std::string > training_detectors;
  std::vector< std::string > training_trackers;

  if( !opt_config.empty() )
  {
    string_to_vector( opt_config, training_configs, "," );
  }
  if( !opt_detector.empty() )
  {
    string_to_vector( opt_detector, training_detectors, "," );
  }
  if( !opt_tracker.empty() )
  {
    string_to_vector( opt_tracker, training_trackers, "," );
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
  kv::plugin_manager::instance().load_all_plugins();
  kv::config_block_sptr config = default_config();
  kv::algo::detected_object_set_input_sptr groundtruth_reader;
  kv::algo::image_io_sptr image_reader;
  kv::algo::train_detector_sptr detector_trainer;
  kv::algo::train_tracker_sptr tracker_trainer;
  kv::algo::read_object_track_set_sptr track_reader;

  // Read all configuration options and check settings (use first config/detector for data loading)
  std::string first_config = training_configs.empty() ? "" : training_configs[0];
  std::string first_detector = training_detectors.empty() ? "" : training_detectors[0];

  if( !first_config.empty() )
  {
    try
    {
      config->merge_config( kv::read_config_file( first_config ) );
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

  if( !opt_settings.empty() )
  {
    const std::string& setting = opt_settings;
    size_t const split_pos = setting.find( "=" );

    if( split_pos == std::string::npos )
    {
      std::string const reason = "Error: The setting on the command line \'"
        + setting + "\' does not contain the \'=\' string which separates "
        "the key from the value";

      throw std::runtime_error( reason );
    }

    kv::config_block_key_t setting_key =
      setting.substr( 0, split_pos );
    kv::config_block_value_t setting_value =
      setting.substr( split_pos + 1 );

    kv::config_block_keys_t keys;

    kv::tokenize( setting_key, keys,
      kv::config_block::block_sep(),
      kv::TokenizeTrimEmpty );

    if( keys.size() < 2 )
    {
      std::string const reason = "Error: The key portion of setting "
        "\'" + setting + "\' does not contain at least two keys in its "
        "keypath which is invalid. (e.g. must be at least a:b)";

      throw std::runtime_error( reason );
    }

    config->set_value( setting_key, setting_value );
  }

  if( opt_no_adv_print )
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
            !opt_timeout.empty() )
        {
          if( !opt_timeout.empty() )
          {
            config->set_value( conf, opt_timeout );
          }
          else
          {
            config->set_value( conf, "1209600" );
          }
        }
      }
    }

    if( opt_no_emb_pipe )
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

  if( !opt_init_weights.empty() )
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

    if( does_folder_exist( opt_init_weights ) )
    {
      for( auto itr : weight_ext )
      {
        std::vector< std::string > files_of_ext;

        list_files_in_folder( opt_init_weights,
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
    else if( does_file_exist( opt_init_weights ) )
    {
      for( auto ext_itr : weight_ext )
      {
        if( ends_with_extension( opt_init_weights, ext_itr.first ) )
        {
          found_files[ ext_itr.first ] = opt_init_weights;
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

  kv::algo::train_detector::set_nested_algo_configuration
    ( "detector_trainer", config, detector_trainer );
  kv::algo::train_detector::get_nested_algo_configuration
    ( "detector_trainer", config, detector_trainer );

  kv::algo::detected_object_set_input::set_nested_algo_configuration
    ( "groundtruth_reader", config, groundtruth_reader );
  kv::algo::detected_object_set_input::get_nested_algo_configuration
    ( "groundtruth_reader", config, groundtruth_reader );

  bool valid_config = true;

  if( !kv::algo::detected_object_set_input::
        check_nested_algo_configuration( "groundtruth_reader", config ) )
  {
    valid_config = false;
  }

  if( !kv::algo::train_detector::
        check_nested_algo_configuration( "detector_trainer", config ) )
  {
    valid_config = false;
  }

  if( !opt_out_config.empty() )
  {
    write_config_file( config, opt_out_config );

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
  std::string output_directory =
    config->get_value< std::string >( "output_directory" );
  std::string output_file =
    config->get_value< std::string >( "output_file" );
  std::string pipeline_template =
    config->get_value< std::string >( "pipeline_template" );
  std::string output_pipeline_name =
    config->get_value< std::string >( "output_pipeline_name" );

  // Command line override for output_file
  if( !opt_output_file.empty() )
  {
    output_file = opt_output_file;
  }

  if( convert_to_full_frame && !kv::algo::image_io::
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

  if( !opt_threshold.empty() )
  {
    threshold = atof( opt_threshold.c_str() );
    std::cout << "Using command line provided threshold: " << threshold << std::endl;
  }

  if( !opt_pipeline_file.empty() )
  {
    pipeline_file = opt_pipeline_file;
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

  if( !opt_frame_rate.empty() )
  {
    frame_rate = std::stod( opt_frame_rate );
  }

  if( !opt_max_frame_count.empty() )
  {
    max_frame_count = std::stoi( opt_max_frame_count );
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

  if( !opt_label_file.empty() )
  {
    label_fn = opt_label_file;
  }
  else if( !opt_input_dir.empty() )
  {
    label_fn = append_path( opt_input_dir, "labels.txt" );
  }

  kv::category_hierarchy_sptr model_labels;
  bool detection_without_label = false;

  if( !does_file_exist( label_fn ) && opt_out_config.empty() )
  {
    std::cout << "Label file (labels.txt) does not exist in input folder" << std::endl;
    std::cout << std::endl << "Would you like to train over all category labels? (y/n) ";

    if( !opt_no_query )
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
  else if( opt_out_config.empty() )
  {
    try
    {
      model_labels.reset( new kv::category_hierarchy( label_fn ) );
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
    kv::algo::image_io::set_nested_algo_configuration
      ( "image_reader", config, image_reader );
    kv::algo::image_io::get_nested_algo_configuration
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
  if( !opt_input_dir.empty() )
  {
    std::string input_dir = resolve_path_with_link( opt_input_dir );

    if( !does_folder_exist( input_dir ) && opt_out_config.empty() )
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
      std::string full_path = append_path( opt_input_dir, to_test );

      if( !does_file_exist( full_path ) && does_file_exist( to_test ) )
      {
        absolute_paths = true;
        std::cout << "Using absolute paths in train.txt and validation.txt" << std::endl;
      }

      for( unsigned i = 0; i < all_data.size(); i++ )
      {
        if( !absolute_paths )
        {
          all_data[i] = append_path( opt_input_dir, all_data[i] );
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
      list_all_subfolders( opt_input_dir, subfolders );
      list_files_in_folder( opt_input_dir, videos, false, video_exts );

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
  else if( !opt_input_list.empty() )
  {
    if( !does_file_exist( opt_input_list ) ||
        !load_file_list( opt_input_list, all_data ) )
    {
      std::cout << "Unable to load: " << opt_input_list << std::endl;
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

    auto_detect_truth = opt_input_truth.empty();

    if( !auto_detect_truth )
    {
      // Check if input_truth is a single file (CSV) or a list file
      if( does_file_exist( opt_input_truth ) )
      {
        // Check if it's a groundtruth file directly (e.g., .csv) or a list file
        bool is_truth_file = ends_with_extension( opt_input_truth, groundtruth_exts );

        if( is_truth_file )
        {
          // Single truth file for all images - replicate it for each data entry
          all_truth.resize( all_data.size(), opt_input_truth );
        }
        else
        {
          // It's a list file containing paths to truth files
          if( !load_file_list( opt_input_truth, all_truth ) )
          {
            std::cout << "Unable to load: " << opt_input_truth << std::endl;
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
        std::cout << "Unable to find: " << opt_input_truth << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  // Load optional manual validation folder
  if( !opt_validation_dir.empty() )
  {
    std::vector< std::string > subfolders, videos;

    if( validation_pivot < 0 )
    {
      validation_pivot = all_data.size();
    }

    if( !does_folder_exist( opt_validation_dir ) )
    {
      std::cerr << "Unable to open " << opt_validation_dir << std::endl;
      return EXIT_FAILURE;
    }

    list_all_subfolders( opt_validation_dir, subfolders );
    list_files_in_folder( opt_validation_dir, videos, false, video_exts );

    all_data.insert( all_data.end(), subfolders.begin(), subfolders.end() );
    all_data.insert( all_data.end(), videos.begin(), videos.end() );
  }

  // Load groundtruth for all image files in all folders using reader class
  std::vector< std::string > train_image_fn;
  std::vector< kv::detected_object_set_sptr > train_gt;
  std::vector< std::string > validation_image_fn;
  std::vector< kv::detected_object_set_sptr > validation_gt;

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
      std::string video_truth = find_associated_file( data_item, groundtruth_exts[0] );

      if( video_truth.empty() )
      {
        std::cout << "Error: cannot find groundtruth for " << data_item << std::endl;
        return EXIT_FAILURE;
      }

      gt_files.resize( 1, video_truth );
    }
    else if( !is_video && auto_detect_truth )
    {
      gt_files = find_files_in_folder_or_alongside( data_item, groundtruth_exts );

      // Handle multiple groundtruth files: allow if different extensions, select by priority
      if( !one_file_per_image && gt_files.size() > 1 )
      {
        std::vector< std::string > priority_exts = { ".csv", ".json", ".xml", ".kw18" };
        std::string selected, error_msg;

        if( !select_file_by_extension_priority(
              gt_files, priority_exts, groundtruth_exts, selected, error_msg ) )
        {
          std::cout << "Error: item " << data_item
                    << " contains " << error_msg << std::endl;
          return EXIT_FAILURE;
        }

        std::cout << "Multiple groundtruth files found, selected: "
                  << get_filename_no_path( selected ) << std::endl;

        gt_files.clear();
        gt_files.push_back( selected );
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
    kv::algo::detected_object_set_input_sptr gt_reader;

    if( !one_file_per_image )
    {
      if( gt_files.size() != 1 )
      {
        std::cout << "Error: item " << data_item
                  << " must contain only 1 groundtruth file" << std::endl;
        return EXIT_FAILURE;
      }

      kv::algo::detected_object_set_input::set_nested_algo_configuration
        ( "groundtruth_reader", config, gt_reader );
      kv::algo::detected_object_set_input::get_nested_algo_configuration
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

    for( unsigned j = 0; j < image_files.size(); ++j )
    {
      const std::string& image_file = image_files[j];

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
      kv::detected_object_set_sptr frame_dets =
        std::make_shared< kv::detected_object_set>();

      if( one_file_per_image )
      {
        gt_reader.reset();

        kv::algo::detected_object_set_input::set_nested_algo_configuration
          ( "groundtruth_reader", config, gt_reader );
        kv::algo::detected_object_set_input::get_nested_algo_configuration
          ( "groundtruth_reader", config, gt_reader );

        gt_reader->open( gt_files[j] );

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
        if( j < 4 || variable_resolution_sequences )
        {
          auto image = image_reader->load( image_file );

          unsigned new_width = image->width();
          unsigned new_height = image->height();

          if( j > 0 && ( new_width != image_width || new_height != image_height ) )
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

        kv::detected_object_set_sptr filtered_dets =
          std::make_shared< kv::detected_object_set>();

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
            kv::detected_object_type_sptr(
              new kv::detected_object_type( label, 1.0 ) ) );

          label_counts[ label ]++;
        }
      }
      for( auto det_set : validation_gt )
      {
        for( auto det : *det_set )
        {
          det->set_type(
            kv::detected_object_type_sptr(
              new kv::detected_object_type( label, 1.0 ) ) );
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
    model_labels.reset( new kv::category_hierarchy() );

    int id = 0;

    for( auto label : label_counts )
    {
      model_labels->add_class( label.first, "", id++ );
    }
  }

  // Use GT frames only if enabled
  if( opt_gt_only )
  {
    std::vector< std::string > adj_train_image_fn;
    std::vector< kv::detected_object_set_sptr > adj_train_gt;

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
    std::vector< kv::detected_object_set_sptr > adj_train_gt;

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
    // Use model-specific config for multi-model training, otherwise use main config
    kv::config_block_sptr current_config = config;

    if( multi_model_training )
    {
      std::cout << std::endl << "========================================" << std::endl;
      std::cout << "Training model " << ( model_idx + 1 ) << " of " << model_count << std::endl;
      std::cout << "========================================" << std::endl;

      // Reconfigure for this model
      current_config = default_config();

      if( !training_configs.empty() )
      {
        std::string model_config_file = training_configs[ model_idx ];
        std::cout << "Using config: " << model_config_file << std::endl;

        try
        {
          current_config->merge_config( kv::read_config_file( model_config_file ) );
        }
        catch( const std::exception& e )
        {
          std::cerr << "Received exception: " << e.what() << std::endl
                    << "Unable to load configuration file: "
                    << model_config_file << std::endl;
          continue;
        }
      }
      else
      {
        std::string current_detector = training_detectors[ model_idx ];
        std::cout << "Using detector type: " << current_detector << std::endl;
        current_config->set_value( "detector_trainer:type", current_detector );
      }

      // Apply command line settings override
      if( !opt_settings.empty() )
      {
        const std::string& setting = opt_settings;
        size_t const split_pos = setting.find( "=" );

        if( split_pos != std::string::npos )
        {
          kv::config_block_key_t setting_key =
            setting.substr( 0, split_pos );
          kv::config_block_value_t setting_value =
            setting.substr( split_pos + 1 );
          current_config->set_value( setting_key, setting_value );
        }
      }

      // Reinitialize detector trainer for this model
      detector_trainer.reset();
      kv::algo::train_detector::set_nested_algo_configuration
        ( "detector_trainer", current_config, detector_trainer );
      kv::algo::train_detector::get_nested_algo_configuration
        ( "detector_trainer", current_config, detector_trainer );

      if( !kv::algo::train_detector::
            check_nested_algo_configuration( "detector_trainer", current_config ) )
      {
        std::cout << "Configuration not valid for model " << ( model_idx + 1 ) << std::endl;
        continue;
      }
    }

    std::cout << "Beginning Training Process" << std::endl;
    std::string error;

    // Get the detector type for validation
    std::string detector_type = current_config->get_value< std::string >(
      "detector_trainer:type", "" );

    try
    {
      detector_trainer->add_data_from_disk( model_labels,
        train_image_fn, train_gt, validation_image_fn, validation_gt );

      std::map< std::string, std::string > trainer_output =
        detector_trainer->update_model();

      process_trainer_output( trainer_output, output_directory, output_file,
        pipeline_template, output_pipeline_name, detector_type, true );
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
    std::vector< kv::object_track_set_sptr > train_tracks;
    std::vector< kv::object_track_set_sptr > validation_tracks;

    // Configure track reader
    kv::algo::read_object_track_set::set_nested_algo_configuration
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
          std::string video_truth = find_associated_file( data_item, groundtruth_exts[0] );
          if( !video_truth.empty() )
          {
            gt_files.push_back( video_truth );
          }
        }
        else if( !is_video && auto_detect_truth )
        {
          gt_files = find_files_in_folder_or_alongside( data_item, groundtruth_exts );
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

            kv::object_track_set_sptr tracks;
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
      kv::config_block_sptr tracker_config = default_config();
      tracker_config->set_value( "tracker_trainer:type", current_tracker );

      // Apply command line settings override
      if( !opt_settings.empty() )
      {
        const std::string& setting = opt_settings;
        size_t const split_pos = setting.find( "=" );

        if( split_pos != std::string::npos )
        {
          kv::config_block_key_t setting_key =
            setting.substr( 0, split_pos );
          kv::config_block_value_t setting_value =
            setting.substr( split_pos + 1 );
          tracker_config->set_value( setting_key, setting_value );
        }
      }

      kv::algo::train_tracker::set_nested_algo_configuration
        ( "tracker_trainer", tracker_config, tracker_trainer );
      kv::algo::train_tracker::get_nested_algo_configuration
        ( "tracker_trainer", tracker_config, tracker_trainer );

      if( !kv::algo::train_tracker::
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

        std::map< std::string, std::string > trainer_output =
          tracker_trainer->update_model();

        process_trainer_output( trainer_output, output_directory, output_file,
          pipeline_template, output_pipeline_name, current_tracker, false );
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

} // namespace tools
} // namespace viame
