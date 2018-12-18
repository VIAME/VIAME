/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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
#include <vital/logger/logger.h>
#include <vital/algo/algorithm_factory.h>
#include <vital/algo/train_detector.h>
#include <vital/algo/detected_object_set_input.h>
#include <vital/config/config_block_io.h>
#include <vital/types/image_container.h>
#include <vital/algo/image_io.h>

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <sprokit/pipeline/process_exception.h>
#include <sprokit/processes/adapters/embedded_pipeline.h>
#include <sprokit/processes/adapters/adapter_types.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <memory>
#include <map>

//==================================================================================================
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
  std::string opt_out_config;
  std::string opt_threshold;
  std::string opt_pipeline_file;

  trainer_vars()
  {
    opt_help = false;
    opt_list = false;
  }

  virtual ~trainer_vars()
  {
  }
};

//==================================================================================================
// Define global variables used across this tool
static trainer_vars g_params;
static kwiver::vital::logger_handle_t g_logger;

//==================================================================================================
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
                           std::vector< std::string >& filepaths,
                           bool search_subfolders = false,
                           std::vector< std::string > extensions = std::vector< std::string >() )
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
      if( extensions.empty() )
      {
        filepaths.push_back( file_iter->path().string() );
      }
      else
      {
        for( unsigned i = 0; i < extensions.size(); i++ )
        {
          if( file_iter->path().extension() == extensions[i] )
          {
            filepaths.push_back( file_iter->path().string() );
            break;
          }
        }
      }
    }
    else if( boost::filesystem::is_directory( *file_iter ) && search_subfolders )
    {
      std::vector< std::string > subfiles;
      list_files_in_folder( file_iter->path().string(), subfiles, search_subfolders, extensions );

      filepaths.insert( filepaths.end(), subfiles.begin(), subfiles.end() );
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

std::string get_filename_with_last_path( std::string path )
{
  return append_path( boost::filesystem::path( path ).parent_path().filename().string(),
                      boost::filesystem::path( path ).filename().string() );
}

std::string get_filename_no_path( std::string path )
{
  return boost::filesystem::path( path ).filename().string();
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
    for( std::string s : parsed_string )
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

void correct_manual_annotations( kwiver::vital::detected_object_set_sptr dos )
{
  if( !dos )
  {
    return;
  }

  for( kwiver::vital::detected_object_sptr do_sptr : *dos )
  {
    if( do_sptr->confidence() < 0.0 )
    {
      do_sptr->set_confidence( 1.0 );
    }

    kwiver::vital::bounding_box_d do_box = do_sptr->bounding_box();

    if( do_box.min_x() > do_box.max_x() )
    {
      do_box = kwiver::vital::bounding_box_d(
        do_box.max_x(), do_box.min_y(), do_box.min_x(), do_box.max_y() );
    }
    if( do_box.min_y() > do_box.max_y() )
    {
      do_box = kwiver::vital::bounding_box_d(
        do_box.min_x(), do_box.max_y(), do_box.max_x(), do_box.min_y());
    }

    do_sptr->set_bounding_box( do_box );

    if( do_sptr->type() )
    {
      kwiver::vital::detected_object_type_sptr type_sptr = do_sptr->type();

      std::string top_category;
      double top_score;

      type_sptr->get_most_likely( top_category, top_score );

      if( top_score < 0.0 )
      {
        type_sptr->set_score( top_category, 1.0 );
        do_sptr->set_type( type_sptr );
      }
    }
  }
}

//==================================================================================================
// Assorted configuration related helper functions
static kwiver::vital::config_block_sptr default_config()
{
  kwiver::vital::config_block_sptr config
    = kwiver::vital::config_block::empty_config( "detector_trainer_tool" );

  config->set_value( "groundtruth_extensions", ".txt",
                     "Groundtruth file extensions (txt, kw18, etc...). Note: this is indepedent "
                     "of the format that's stored in the file" );
  config->set_value( "groundtruth_style", "one_per_folder",
                     "Can be either: \"one_per_file\" or \"one_per_folder\"" );

  config->set_value( "default_percent_test", "0.05",
                     "Percent [0.0, 1.0] of test samples to use if no manual files specified." );
  config->set_value( "image_extensions", ".jpg;.jpeg;.JPG;.JPEG;.tif;.tiff;.TIF;.TIFF;.png;.PNG",
                     "Semicolon list of seperated image extensions to use in training, images "
                     "without this extension will not be included." );
  config->set_value( "threshold", "0.00",
                     "Optional threshold to provide on top of input groundtruth. This is useful "
                     "if the groundtruth is derived from some automated detector." );
  config->set_value( "check_override", "false",
                     "Over-ride and ignore data safety checks." );

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

std::unique_ptr<kwiver::embedded_pipeline>
  get_embedded_pipeline(std::string &pipeline_filename) {
  std::unique_ptr< kwiver::embedded_pipeline > external_pipeline;
  auto dir = boost::filesystem::path(pipeline_filename).parent_path();

  if(!pipeline_filename.empty())
  {
    std::unique_ptr<kwiver::embedded_pipeline> new_pipeline =
        std::unique_ptr<kwiver::embedded_pipeline>(new kwiver::embedded_pipeline());

    std::ifstream pipe_stream;
    pipe_stream.open(pipeline_filename, std::ifstream::in );

    if(!pipe_stream)
    {
      throw sprokit::invalid_configuration_exception("viame_train_detector",
          "Unable to open pipeline file: " + pipeline_filename);
    }

    try
    {
      new_pipeline->build_pipeline(pipe_stream, dir.string());
      new_pipeline->start();
    }
    catch(const std::exception& e)
    {
      throw sprokit::invalid_configuration_exception("viame_train_detector",
                                                     e.what());
    }

    external_pipeline = std::move(new_pipeline);
    pipe_stream.close();
  }
  return external_pipeline;
}

kwiver::vital::image_container_sptr load_image(std::string image_name) {

  kwiver::vital::algo::image_io_sptr image_reader =
      kwiver::vital::algo::image_io::create("ocv");

  kwiver::vital::image_container_sptr
      the_image = image_reader->load(image_name);

  return the_image;
}

kwiver::adapter::adapter_data_set_t
get_ids_for_split_image_pipe_line(std::string image_name) {
  kwiver::adapter::adapter_data_set_t ids =
      kwiver::adapter::adapter_data_set::create();
  kwiver::vital::image_container_sptr the_image = load_image(image_name);
  ids->add_value("image", the_image);
  return ids;
}

std::string get_modified_image_name(std::string name) {

  std::string parent_directory =
      kwiversys::SystemTools::GetParentDirectory(name);

  std::string file_name =
      kwiversys::SystemTools::GetFilenameWithoutExtension(name);

  std::string last_extension =
      kwiversys::SystemTools::GetFilenameLastExtension(name);

  std::vector<std::string> full_path;
  boost::filesystem::path p = boost::filesystem::temp_directory_path();

  full_path.push_back("");
  full_path.push_back(p.string());
  full_path.push_back(file_name + last_extension);

  std::string mod_path = kwiversys::SystemTools::JoinPath(full_path);
  std::cout << "mod_path = " << mod_path << std::endl;
  return mod_path;
}

kwiver::vital::image_container_sptr
run_pipeline_on_image(std::string image_name, std::string opt_pipeline) {

  std::unique_ptr <kwiver::embedded_pipeline> external_pipeline =
      get_embedded_pipeline(opt_pipeline);

  kwiver::adapter::adapter_data_set_t ids =
      kwiver::adapter::adapter_data_set::create();

  kwiver::vital::image_container_sptr image_sent = load_image(image_name);

  ids->add_value("image", image_sent);

  external_pipeline->send(ids);
  external_pipeline->send_end_of_input();

  auto const &ods = external_pipeline->receive();
  external_pipeline->wait();
  external_pipeline.reset();

  if (ods->is_end_of_data()) {
    throw std::runtime_error("Pipeline terminated unexpectingly");
  }
  auto const &image_received = ods->find("image");

  kwiver::vital::image_container_sptr image_sptr =
      image_received->second->get_datum<kwiver::vital::image_container_sptr>();

  return image_sptr;
}

// =================================================================================================
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
  g_params.m_args.AddArgument( "--output-config", argT::SPACE_ARGUMENT,
    &g_params.opt_out_config, "Output a sample configuration to file" );
  g_params.m_args.AddArgument( "-o",        argT::SPACE_ARGUMENT,
    &g_params.opt_out_config, "Output a sample configuration to file" );
  g_params.m_args.AddArgument( "--threshold", argT::SPACE_ARGUMENT,
    &g_params.opt_threshold, "Threshold override to apply over inputs" );
  g_params.m_args.AddArgument( "-t",        argT::SPACE_ARGUMENT,
    &g_params.opt_threshold, "Threshold override to apply over inputs" );
  g_params.m_args.AddArgument( "--pipeline", argT::SPACE_ARGUMENT,
                               &g_params.opt_pipeline_file, "Pipeline file" );
  g_params.m_args.AddArgument( "-p", argT::SPACE_ARGUMENT,
                               &g_params.opt_pipeline_file, "Pipeline file" );

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

  if( !does_folder_exist( input_dir ) && g_params.opt_out_config.empty() )
  {
    std::cerr << "Input directory does not exist, exiting." << std::endl;
    exit( 0 );
  }

  // Load labels.txt file
  const std::string label_fn = append_path( input_dir, "labels.txt" );

  kwiver::vital::category_hierarchy_sptr classes;

  if( !does_file_exist( label_fn ) && g_params.opt_out_config.empty() )
  {
    std::cerr << "Label file (label.txt) does not exist in data folder" << std::endl;
    exit( 0 );
  }
  else if( g_params.opt_out_config.empty() )
  {
    classes.reset( new kwiver::vital::category_hierarchy( label_fn ) );
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
        std::cerr << "Could not find test file: " << test_files[i] << std::endl;
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
    try
    {
      config->merge_config(kwiver::vital::read_config_file(g_params.opt_config));
    }
    catch( const std::exception& e )
    {
      std::cerr << "Received exception: " << e.what() << std::endl;
      std::cerr << "Unable to load configuration file: " << g_params.opt_config << std::endl;
      exit( 0 );
    }
  }
  else
  {
    config->set_value( "detector_trainer:type", g_params.opt_detector );
  }

  kwiver::vital::algo::train_detector::set_nested_algo_configuration
    ( "detector_trainer", config, detector_trainer );
  kwiver::vital::algo::train_detector::get_nested_algo_configuration
    ( "detector_trainer", config, detector_trainer );

  kwiver::vital::algo::detected_object_set_input::set_nested_algo_configuration
    ( "groundtruth_reader", config, groundtruth_reader );
  kwiver::vital::algo::detected_object_set_input::get_nested_algo_configuration
    ( "groundtruth_reader", config, groundtruth_reader );

  bool valid_config = check_config( config );

  if( !g_params.opt_out_config.empty() )
  {
    write_config_file( config, g_params.opt_out_config );

    if( valid_config )
    {
      std::cout << "Configuration file contained valid parameters "
        "and may be used for running" << std::endl;
    }
    else
    {
      std::cout << "Configuration deemed not valid." << std::endl;
    }
    return EXIT_SUCCESS;
  }
  else if( !valid_config )
  {
    std::cout << "Configuration not valid." << std::endl;
    return EXIT_FAILURE;
  }

  // Read setup configs
  double percent_test = config->get_value< double >( "default_percent_test" );
  std::string groundtruth_extensions_str = config->get_value< std::string >( "groundtruth_extensions" );
  std::string image_extensions_str = config->get_value< std::string >( "image_extensions" );
  std::string groundtruth_style = config->get_value< std::string >( "groundtruth_style" );
  bool check_override = config->get_value< bool >( "check_override" );
  double threshold = config->get_value< double >( "threshold" );

  if( !g_params.opt_threshold.empty() )
  {
    threshold = atof( g_params.opt_threshold.c_str() );
    std::cout << "Using command line provided threshold: " << threshold << std::endl;
  }

  std::vector< std::string > image_extensions, groundtruth_extensions;
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
  string_to_vector( groundtruth_extensions_str, groundtruth_extensions, "\n\t\v,; " );

  // Identify all sub-directories containing data
  std::vector< std::string > subdirs;
  list_all_subfolders( g_params.opt_input, subdirs );

  // Load groundtruth for all image files in all folders using reader class
  std::vector< std::string > train_image_fn;
  std::vector< kwiver::vital::detected_object_set_sptr > train_gt;
  std::vector< std::string > test_image_fn;
  std::vector< kwiver::vital::detected_object_set_sptr > test_gt;

  if( subdirs.empty() )
  {
    std::cout << "Error: training folder contains no sub-folders" << std::endl;
    exit( 0 );
  }

  // Retain class counts for error checking
  std::map< std::string, int > class_count;

  for( std::string folder : subdirs )
  {
    std::cout << "Processing " << folder << std::endl;

    std::string fullpath = folder;

    std::vector< std::string > image_files, gt_files;
    std::vector< std::string > image_files_after_pipeline;
    list_files_in_folder( fullpath, image_files, true, image_extensions );
    list_files_in_folder( fullpath, gt_files, false, groundtruth_extensions );

    std::sort( image_files.begin(), image_files.end() );
    std::sort( gt_files.begin(), gt_files.end() );

    if( one_file_per_image && ( image_files.size() != gt_files.size() ) )
    {
      std::cout << "Error: folder " << folder << " contains unequal truth and image file counts" << std::endl;
      std::cout << " - Consider turning on the one_per_folder groundtruth style" << std::endl;
      exit( 0 );
    }
    else if( gt_files.size() < 1 )
    {
      std::cout << "Error reading folder " << folder << ", no groundtruth." << std::endl;
      exit( 0 );
    }

    kwiver::vital::algo::detected_object_set_input_sptr gt_reader;

    if( !one_file_per_image )
    {
      if( gt_files.size() != 1 )
      {
        std::cout << "Error: folder " << folder << " must contain only 1 groundtruth file" << std::endl;
        exit( 0 );
      }

      kwiver::vital::algo::detected_object_set_input::set_nested_algo_configuration
        ( "groundtruth_reader", config, gt_reader );
      kwiver::vital::algo::detected_object_set_input::get_nested_algo_configuration
        ( "groundtruth_reader", config, gt_reader );

      std::cout << "Opening groundtruth file " << gt_files[0] << std::endl;

      gt_reader->open( gt_files[0] );
    }

    // Read all images and detections in sequence
    if( image_files.size() == 0 )
    {
      std::cout << "Error: folder contains no image files." << std::endl;
    }

    for( unsigned i = 0; i < image_files.size(); ++i )
    {
      const std::string image_file = image_files[i];

      std::string opt_pipeline = g_params.opt_pipeline_file;
      kwiver::vital::image_container_sptr image_sptr =
          run_pipeline_on_image(image_file, opt_pipeline);


      const std::string image_file_after_pipeline =
          get_modified_image_name(image_file);
      image_files_after_pipeline.push_back(image_file_after_pipeline);

      kwiver::vital::algo::image_io_sptr
          image_writer = kwiver::vital::algo::image_io::create("ocv");

      image_writer->save(image_file_after_pipeline, image_sptr);

      const std::string file_wrt_input = append_path( folder,
                                                      image_file_after_pipeline );
      const std::string file_full_path = append_path( g_params.opt_input, file_wrt_input );

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

        std::string read_fn = get_filename_no_path( image_file_after_pipeline );
        gt_reader->read_set( frame_dets, read_fn );
        gt_reader->close();

        correct_manual_annotations( frame_dets );
      }
      else
      {
        std::string read_fn = get_filename_no_path( image_file_after_pipeline );
        try
        {
          gt_reader->read_set( frame_dets, read_fn );

          correct_manual_annotations( frame_dets );
        }
        catch( const std::exception& e )
        {
          std::cerr << "Received exception: " << e.what() << std::endl;
          std::cerr << "Unable to load groundtruth file: " << read_fn << std::endl;
          exit( 0 );
        }
      }

      std::cout << "Read " << frame_dets->size() << " detections for " << image_file_after_pipeline << std::endl;

      // Apply threshold to frame detections
      kwiver::vital::detected_object_set_sptr filtered_dets =
        std::make_shared< kwiver::vital::detected_object_set>();

      for( auto det : *frame_dets )
      {
        bool add_detection = false;

        if( det->type() )
        {
          for( auto t : *det->type() )
          {
            std::string gt_class = *(t.first);

            if( classes->has_class_name( gt_class ) )
            {
              if( t.second > threshold )
              {
                class_count[classes->get_class_name( gt_class )]++;
                add_detection = true;
              }
            }
            else
            {
              det->type()->delete_score( gt_class );
            }
          }
        }
        else if( classes->size() == 1 )
        {
          add_detection = true; // single class problem, doesn't need dot
        }

        if( add_detection )
        {
          filtered_dets->add( det );
        }
      }

      // TODO: Is this a train or test image?
      train_image_fn.push_back( image_files_after_pipeline[i] );
      train_gt.push_back( filtered_dets );
    }

    if( !one_file_per_image )
    {
      gt_reader->close();
    }
  }

  for( auto det_set : train_gt )
  {
    for( auto det : *det_set )
    {
      if( det->type() )
      {
        std::string gt_class;
        det->type()->get_most_likely( gt_class );

        if( classes->has_class_name( gt_class ) )
        {
          class_count[ classes->get_class_name( gt_class ) ]++;
        }
      }
    }
  }

  if( class_count.empty() ) // groundtruth has no classification labels
  {
    // Only 1 class, is okay but inject the classification into the groundtruth
    if( classes->size() == 1 )
    {
      for( auto det_set : train_gt )
      {
        for( auto det : *det_set )
        {
          det->set_type(
            kwiver::vital::detected_object_type_sptr(
              new kwiver::vital::detected_object_type( classes->all_class_names()[0], 1.0 ) ) );
        }
      }
      for( auto det_set : test_gt )
      {
        for( auto det : *det_set )
        {
          det->set_type(
            kwiver::vital::detected_object_type_sptr(
              new kwiver::vital::detected_object_type( classes->all_class_names()[0], 1.0 ) ) );
        }
      }
    }
    else // Not okay
    {
      std::cout << "Error: labels.txt contains multiple classes, but GT is does "
                   "not contain classes of interest." << std::endl;
      return 0;
    }
  }
  else if( !check_override )
  {
    for( auto cls : classes->all_class_names() )
    {
      if( class_count[ cls ] == 0 )
      {
        std::cout << "Error: no entries in groundtruth of class " << cls << std::endl;
        std::cout << "Optionally set \"check_override\" parameter to ignore this check." << std::endl;
        exit( 0 );
      }
    }
  }

  // Run training algorithm
  std::cout << "Beginning Training Process" << std::endl;

  detector_trainer->train_from_disk( classes, train_image_fn, train_gt, test_image_fn, test_gt );
  return 0;
}
