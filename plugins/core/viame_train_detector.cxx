/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <kwiversys/SystemTools.hxx>
#include <kwiversys/CommandLineArguments.hxx>

#include <vital/algorithm_plugin_manager_paths.h>

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/config/config_block.h>
#include <vital/config/config_block_io.h>
#include <vital/util/demangle.h>
#include <vital/util/wrap_text_block.h>
#include <vital/vital_foreach.h>
#include <vital/logger/logger.h>
#include <vital/algo/algorithm_factory.h>
#include <vital/algo/train_detector.h>
#include <vital/algo/detected_object_set_input.h>

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <memory>
#include <map>

//===================================================================
// Class storing all input parameters and private variables for tool
class trainer_vars
{
public:

  // Collected command line args
  kwiversys::CommandLineArguments m_args;

  // Config options
  bool opt_help;
  bool opt_list;
  std::string opt_config;
  std::string opt_input;
  std::string opt_detector;

  trainer_vars()
  {
    opt_help = false;
    opt_list = false;
  }

  virtual ~trainer_vars()
  {
  }
};

//===================================================================
// Define global variables used across this tool
static trainer_vars g_params;
static kwiver::vital::logger_handle_t g_logger;

//===================================================================
// Assorted filesystem related helper functions
bool does_file_exist( const std::string& location )
{
  return boost::filesystem::exists( location ) &&
         !boost::filesystem::is_directory( location );
}

bool does_folder_exist( const std::string& location )
{
  return boost::filesystem::exists( location ) &&
         boost::filesystem::is_directory( location );
}

bool list_all_subfolders( const std::string& location,
                          std::vector< std::string >& subfolders )
{
  subfolders.clear();

  if( !does_folder_exist( location ) )
  {
    return false;
  }

  boost::filesystem::path dir( location );

  for( boost::filesystem::directory_iterator dir_iter( dir );
       dir_iter != boost::filesystem::directory_iterator();
       ++dir_iter )
  {
    if( boost::filesystem::is_directory( *dir_iter ) )
    {
      subfolders.push_back( dir_iter->path().string() );
    }
  }

  return true;
}

bool list_files_in_folder( const std::string& location,
                           std::vector< std::string >& filepaths )
{
  filepaths.clear();

  if( !does_folder_exist( location ) )
  {
    return false;
  }

  boost::filesystem::path dir( location );

  for( boost::filesystem::directory_iterator file_iter( dir );
       file_iter != boost::filesystem::directory_iterator();
       ++file_iter )
  {
    if( boost::filesystem::is_regular_file( *file_iter ) )
    {
      filepaths.push_back( file_iter->path().string() );
    }
  }

  return true;
}

bool create_folder( const std::string& location )
{
  boost::filesystem::path dir( location );

  return boost::filesystem::create_directories( dir );
}

std::string append_path( std::string p1, std::string p2 )
{
  return p1 + "/" + p2;
}

bool remove_and_reset_folder( std::string location )
{
  if( does_folder_exist( location ) )
  {
    boost::filesystem::remove_all( location );
  }

  create_folder( location );
  return true;
}

template< typename T >
bool string_to_vector( const std::string& str,
                       std::vector< T >& out,
                       const std::string delims = "\n\t\v ," )
{
  out.clear();

  std::vector< std::string > parsed_string;

  boost::split( parsed_string, str,
                boost::is_any_of( delims ),
                boost::token_compress_on );

  try
  {
    VITAL_FOREACH( std::string s, parsed_string )
    {
      if( !s.empty() )
      {
        out.push_back( boost::lexical_cast< T >( s ) );
      }
    }
  }
  catch( boost::bad_lexical_cast& )
  {
    return false;
  }

  return true;
}

template< typename T >
bool file_to_vector( const std::string& fn, std::vector< T >& out )
{
  std::ifstream in( fn.c_str() );
  out.clear();

  if( !in )
  {
    std::cerr << "Unable to open " << fn << std::endl;
    return false;
  }

  std::string line;
  while( std::getline( in, line ) )
  {
    if( !line.empty() )
    {
      out.push_back( boost::lexical_cast< T >( line ) );
    }
  }
  return true;
}

//===================================================================
// Assorted configuration related helper functions
static kwiver::vital::config_block_sptr default_config()
{
  kwiver::vital::config_block_sptr config
    = kwiver::vital::config_block::empty_config( "detector_trainer_tool" );

  config->set_value( "groundtruth_extension", "txt",
                     "Groundtruth file extension (txt, kw18, etc...)" );
  config->set_value( "groundtruth_style", "one_per_file",
                     "Can be either: \"one_per_file\" or \"one_per_folder\"" );

  config->set_value( "default_percent_test", "0.05",
                     "Percent [0.0, 1.0] of test samples to use if no manual files specified." );
  config->set_value( "image_extensions", "jpg;jpeg;JPG;JPEG;tif;tiff;TIF;TIFF;png;PNG",
                     "Semicolon list of seperated image extensions to use in training if no "
                     "manual files specified." );

  kwiver::vital::algo::detected_object_set_input::get_nested_algo_configuration
    ( "groundtruth_reader", config, kwiver::vital::algo::detected_object_set_input_sptr() );
  kwiver::vital::algo::train_detector::get_nested_algo_configuration
    ( "detector_trainer", config, kwiver::vital::algo::train_detector_sptr() );

  return config;
}

static bool check_config( kwiver::vital::config_block_sptr config )
{
  if( !kwiver::vital::algo::detected_object_set_input::
        check_nested_algo_configuration( "groundtruth_reader", config ) )
  {
    return false;
  }

  if( !kwiver::vital::algo::train_detector::
        check_nested_algo_configuration( "detector_trainer", config ) )
  {
    return false;
  }

  return true;
}

// ==================================================================
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

  g_params.m_args.AddArgument( "--help",    argT::NO_ARGUMENT,
    &g_params.opt_help, "Display usage information" );
  g_params.m_args.AddArgument( "-h",        argT::NO_ARGUMENT,
    &g_params.opt_help, "Display usage information" );
  g_params.m_args.AddArgument( "--list",    argT::NO_ARGUMENT,
    &g_params.opt_list, "Display list of all trainable algorithms" );
  g_params.m_args.AddArgument( "-l",        argT::NO_ARGUMENT,
    &g_params.opt_list, "Display list of all trainable algorithms" );
  g_params.m_args.AddArgument( "--config",  argT::SPACE_ARGUMENT,
    &g_params.opt_config, "Input configuration file with parameters" );
  g_params.m_args.AddArgument( "-c",        argT::SPACE_ARGUMENT,
    &g_params.opt_config, "Input configuration file with parameters" );
  g_params.m_args.AddArgument( "--input",   argT::SPACE_ARGUMENT,
    &g_params.opt_input, "Input directory containing groundtruth" );
  g_params.m_args.AddArgument( "-i",        argT::SPACE_ARGUMENT,
    &g_params.opt_input, "Input directory containing groundtruth" );
  g_params.m_args.AddArgument( "--detector",argT::SPACE_ARGUMENT,
    &g_params.opt_detector, "Type of detector to train if no config" );
  g_params.m_args.AddArgument( "-d",        argT::SPACE_ARGUMENT,
    &g_params.opt_detector, "Type of detector to train if no config" );

  // Parse args
  if( !g_params.m_args.Parse() )
  {
    std::cerr << "Problem parsing arguments" << std::endl;
    exit( 0 );
  }

  // Print help
  if( argc == 1 || g_params.opt_help )
  {
    std::cout << "Usage: " << argv[0] << "[options]\n"
              << "\nTrain one of several object detectors in the system.\n"
              << g_params.m_args.GetHelp() << std::endl;
    exit( 0 );
  }

  // List option
  if( g_params.opt_list )
  {
    kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

    kwiver::vital::path_list_t pathl;
    const std::string& default_module_paths( DEFAULT_MODULE_PATHS );

    kwiversys::SystemTools::Split( default_module_paths, pathl, PATH_SEPARATOR_CHAR );

    VITAL_FOREACH( auto path, pathl )
    {
      vpm.add_search_path( path );
    }

    vpm.load_plugins( pathl );

    auto fact_list = vpm.get_factories(  "train_detector" );

    if( fact_list.empty() )
    {
      std::cerr << "No loaded detectors to list" << std::endl;
    }
    else
    {
      std::cout << std::endl << "Trainable detector variants:" << std::endl << std::endl;
    }

    VITAL_FOREACH( auto fact, fact_list )
    {
      std::string name;
      if( fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, name ) )
      {
        std::cout << name << std::endl;
      }
    }
    exit( 0 );
  }

  // Test for presence of two configs
  if( !g_params.opt_config.empty() && !g_params.opt_detector.empty() )
  {
    std::cerr << "Only one of --config and --detector allowed." << std::endl;
    exit( 0 );
  }

  // Test for presence of two configs
  if( g_params.opt_config.empty() && g_params.opt_detector.empty() )
  {
    std::cerr << "One of --config and --detector must be set." << std::endl;
    exit( 0 );
  }

  // Run tool:
  //   (a) Load groundtruth according to criteria
  //   (b) Select detector to train

  std::string input_dir = g_params.opt_input;

  if( !does_folder_exist( input_dir ) )
  {
    std::cerr << "Input directory does not exist, exiting." << std::endl;
    exit( 0 );
  }

  // Load labels.txt file
  const std::string label_fn = append_path( input_dir, "labels.txt" );

  std::vector< std::string > labels;
  std::vector< std::vector< std::string > > label_ids;

  if( !does_file_exist( label_fn ) )
  {
    std::cerr << "Label file does not exist" << std::endl;
    exit( 0 );
  }
  else
  {
    std::ifstream in( label_fn.c_str() );

    if( !in )
    {
      std::cerr << "Unable to open " << label_fn << std::endl;
      exit( 0 );
    }

    std::string line, label;
    while( std::getline( in, line ) )
    {
      std::vector< std::string > tokens;
      string_to_vector( line, tokens, "\n\t\v " );

      if( tokens.size() == 0 )
      {
        continue;
      }
      else
      {
        std::vector< std::string > id_strs;
        string_to_vector( line, id_strs, "\n\t\v," );
        labels.push_back( tokens[0] );
        label_ids.push_back( id_strs );
      }
    }
  }

  // Load train.txt, if available
  const std::string train_fn = append_path( input_dir, "train.txt" );

  std::vector< std::string > train_files;
  if( does_file_exist( train_fn ) && !file_to_vector( train_fn, train_files ) )
  {
    std::cerr << "Unable to open " << label_fn << std::endl;
    exit( 0 );
  }

  // Load test.txt, if available
  const std::string test_fn = append_path( input_dir, "test.txt" );

  std::vector< std::string > test_files;
  if( does_file_exist( test_fn ) && !file_to_vector( test_fn, test_files ) )
  {
    std::cerr << "Unable to open " << test_fn << std::endl;
    exit( 0 );
  }

  // Append path to all test and train files, test to see if they all exist
  if( train_files.empty() && test_files.empty() )
  {
    std::cout << "Automatically selecting train and test files" << std::endl;
  }
  else if( train_files.empty() != test_files.empty() )
  {
    std::cerr << "If one of either train.txt or test.txt is specified, "
      "then they must both be." << std::endl;
    exit( 0 );
  }
  else
  {
    // Test first entry
    bool absolute_paths = false;
    std::string to_test = train_files[0];
    std::string full_path = append_path( g_params.opt_input, to_test );

    if( !does_file_exist( full_path ) && does_file_exist( to_test ) )
    {
      absolute_paths = true;
      std::cout << "Using absolute paths in train.txt and test.txt" << std::endl;
    }

    for( unsigned i = 0; i < train_files.size(); i++ )
    {
      if( !absolute_paths )
      {
        train_files[i] = append_path( g_params.opt_input, train_files[i] );
      }

      if( !does_file_exist( train_files[i] ) )
      {
        std::cerr << "Could not find train file: " << train_files[i] << std::endl;
      }
    }
    for( unsigned i = 0; i < test_files.size(); i++ )
    {
      if( !absolute_paths )
      {
        test_files[i] = append_path( g_params.opt_input, test_files[i] );
      }

      if( !does_file_exist( test_files[i] ) )
      {
        std::cerr << "Could not find train file: " << test_files[i] << std::endl;
      }
    }
  }

  // Identify technique to run and parse config if it's available
  kwiver::vital::plugin_manager::instance().load_all_plugins();
  kwiver::vital::config_block_sptr config = default_config();
  kwiver::vital::algo::detected_object_set_input_sptr groundtruth_reader;
  kwiver::vital::algo::train_detector_sptr detector_trainer;

  if( !g_params.opt_config.empty() )
  {
    config->merge_config( kwiver::vital::read_config_file( g_params.opt_config ) );
  }
  else
  {
    config->set_value( "detector_trainer_tool:detector_trainer:type", g_params.opt_detector );
  }

  kwiver::vital::algo::detected_object_set_input::get_nested_algo_configuration
    ( "detected_object_set_input", config, groundtruth_reader );
  kwiver::vital::algo::detected_object_set_input::set_nested_algo_configuration
    ( "detected_object_set_input", config, groundtruth_reader );

  kwiver::vital::algo::train_detector::get_nested_algo_configuration
    ( "detector_trainer", config, detector_trainer );
  kwiver::vital::algo::train_detector::set_nested_algo_configuration
    ( "detector_trainer", config, detector_trainer );

  // Read setup configs
  double percent_test = config->get_value< double >( "default_percent_test" );
  std::string groundtruth_extension = config->get_value< std::string >( "groundtruth_extension" );
  std::string image_extensions_str = config->get_value< std::string >( "image_extensions" );
  std::string groundtruth_style = config->get_value< std::string >( "groundtruth_style" );

  std::vector< std::string > image_extensions;
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
    exit( 0 );
  }

  if( percent_test < 0.0 || percent_test > 1.0 )
  {
    std::cerr << "Percent test must be [0.0,1.0]" << std::endl;
    exit( 0 );
  }

  string_to_vector( image_extensions_str, image_extensions, "\n\t\v,; " );

  // Identify all sub-directories containing data
  std::vector< std::string > subdirs;
  list_all_subfolders( g_params.opt_input, subdirs );

  // Load groundtruth for all image files in all folders using reader class
  std::vector< std::string > train_image_fn;
  std::vector< kwiver::vital::detected_object_set_sptr > train_gt;
  std::vector< std::string > test_image_fn;
  std::vector< kwiver::vital::detected_object_set_sptr > test_gt;

  VITAL_FOREACH( std::string folder, subdirs )
  {
    std::string fullpath = append_path( g_params.opt_input, folder );

    std::vector< std::string > files;
    list_files_in_folder( fullpath, files );
    std::sort( files.begin(), files.end() );

    VITAL_FOREACH( std::string file, files )
    {
      
    }
  }

  // Run training algorithm
  detector_trainer->train_from_disk( train_image_fn, train_gt, test_image_fn, test_gt );
  return 0;
}
