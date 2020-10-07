/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#include "init_cameras_landmarks.h"

#include <kwiversys/CommandLineArguments.hxx>
#include <kwiversys/SystemTools.hxx>

#include <vital/algo/initialize_cameras_landmarks.h>
#include <vital/algo/video_input.h>
#include <vital/config/config_block_io.h>
#include <vital/config/config_block.h>
#include <vital/config/config_parser.h>
#include <vital/io/camera_io.h>
#include <vital/io/landmark_map_io.h>
#include <vital/io/track_set_io.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/util/get_paths.h>

#include <fstream>
#include <iostream>

namespace kwiver {
namespace arrows {
namespace mvg {

using kwiver::vital::feature_track_set_sptr;
using kwiver::vital::algo::initialize_cameras_landmarks;
using kwiver::vital::algo::initialize_cameras_landmarks_sptr;
using kwiver::vital::algo::video_input;
using kwiver::vital::algo::video_input_sptr;
using kwiver::vital::camera_map_sptr;
using kwiver::vital::camera_perspective;
using kwiver::vital::camera_sptr;
using kwiver::vital::landmark_map_sptr;
using kwiver::vital::sfm_constraints;
using kwiver::vital::sfm_constraints_sptr;




namespace {

typedef kwiversys::SystemTools ST;
typedef kwiversys::CommandLineArguments argT;

kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( "init_cameras_landmarks" ) );

// ------------------------------------------------------------------
kwiver::vital::config_block_sptr default_config()
{
  kwiver::vital::config_block_sptr config =
    kwiver::vital::config_block::empty_config("init_cameras_landmarks");

  config->set_value("video_source", "",
                    "(optional) Path to an input file to be opened as a video. "
                    "This could be either a video file or a text file "
                    "containing new-line separated paths to sequential "
                    "image files.");
  config->set_value("input_tracks_file", "",
                    "Path to a file to read input tracks from.");
  config->set_value("output_cameras_director", "",
                    "Directory to write cameras to.");
  config->set_value("camera_prefix", "camera",
                    "The prefix to the camera name.");
  config->set_value("output_landmarks_filename", "",
                    "Path to a file to write output landmarks to. If this "
                    "file exists, it will be overwritten.");


  initialize_cameras_landmarks::get_nested_algo_configuration("initializer",
                                                              config,
                                                              initialize_cameras_landmarks_sptr());
  video_input::get_nested_algo_configuration("video_reader", config,
                                             video_input_sptr());
  return config;
}

// ------------------------------------------------------------------
bool check_config(kwiver::vital::config_block_sptr config)
{
  bool config_valid = true;

#define KWIVER_CONFIG_FAIL(msg) \
  LOG_ERROR(main_logger, "Config Check Fail: " << msg); \
  config_valid = false

  if(config->has_value("video_source") &&
    config->get_value<std::string>("video_source") != "")
  {
    std::string path = config->get_value<std::string>("video_source");
    if ( ! ST::FileExists( kwiver::vital::path_t(path), true ) )
    {
      KWIVER_CONFIG_FAIL("video_source path, " << path
                         << ", does not exist or is not a regular file");
    }
    if ( !video_input::check_nested_algo_configuration("video_reader", config) )
    {
      KWIVER_CONFIG_FAIL("video_reader configuration check failed");
    }
  }

  if ( ! config->has_value("input_tracks_file") ||
    config->get_value<std::string>("input_tracks_file") == "")
  {
    KWIVER_CONFIG_FAIL("Config needs value input_tracks_file");
  }
  else
  {
    std::string path = config->get_value<std::string>("input_tracks_file");
    if ( ! ST::FileExists( kwiver::vital::path_t(path), true ) )
    {
      KWIVER_CONFIG_FAIL("input_tracks_file path, " << path
                         << ", does not exist or is not a regular file");
    }
  }

  if (!config->has_value("output_cameras_director") ||
    config->get_value<std::string>("output_cameras_director") == "" )
  {
    KWIVER_CONFIG_FAIL("Config needs value output_cameras_director");
  }
  else if ( ! ST::FileIsDirectory(
    config->get_value<kwiver::vital::path_t>("output_cameras_director") ) )
  {
    KWIVER_CONFIG_FAIL("output_cameras_director is not a valid directory");
  }

  if (!config->has_value("output_landmarks_filename") ||
    config->get_value<std::string>("output_landmarks_filename") == "" )
  {
    KWIVER_CONFIG_FAIL("Config needs value output_landmarks_filename");
  }
  else if ( ! ST::FileIsDirectory( ST::CollapseFullPath( ST::GetFilenamePath(
    config->get_value<kwiver::vital::path_t>("output_landmarks_filename") ))))
  {
    KWIVER_CONFIG_FAIL("output_landmarks_filename is not in a valid directory");
  }

#undef KWIVER_CONFIG_FAIL

  return config_valid;
}
} // end namespace

class init_cameras_landmarks::priv
{
public:
  priv()
  : camera_map_ptr(nullptr),
    landmark_map_ptr(nullptr),
    feature_track_set_ptr(nullptr),
    sfm_constraint_ptr(nullptr),
    config(nullptr),
    num_frames(0)
  {}

  camera_map_sptr camera_map_ptr;
  landmark_map_sptr landmark_map_ptr;
  feature_track_set_sptr feature_track_set_ptr;
  sfm_constraints_sptr sfm_constraint_ptr;
  initialize_cameras_landmarks_sptr algorithm;
  kwiver::vital::config_block_sptr config;
  size_t num_frames;

  enum commandline_mode {SUCCESS, HELP, WRITE, FAIL};

  commandline_mode process_command_line(cxxopts::ParseResult& cmd_args)
  {
    static std::string opt_config;
    static std::string opt_out_config;

    if ( cmd_args["help"].as<bool>() )
    {
      return HELP;
    }
    if ( cmd_args.count("config") )
    {
      opt_config = cmd_args["config"].as<std::string>();
    }
    if ( cmd_args.count("output-config") > 0 )
    {
      opt_out_config = cmd_args["output-config"].as<std::string>();
    }

    // Set up top level configuration w/ defaults where applicable.
    config = default_config();

    // If -c/--config given, read in confg file, merge in with default just
    // generated
    if( ! opt_config.empty() )
    {
      config->merge_config(kwiver::vital::read_config_file(opt_config));
    }

    bool valid_config = check_config(config);

    if( ! opt_out_config.empty() )
    {
      write_config_file(config, opt_out_config );
      if(valid_config)
      {
        LOG_INFO(main_logger,
                 "Configuration file contained valid parameters and may be "
                 "used for running");
      }
      else
      {
        LOG_WARN(main_logger, "Configuration deemed not valid.");
      }
      config = nullptr;
      return WRITE;
    }
    else if(!valid_config)
    {
      LOG_ERROR(main_logger, "Configuration not valid.");
      config = nullptr;
      return FAIL;
    }

    return SUCCESS;
  }

  void register_algorithms()
  {
    // register the algorithm implementations
    std::string rel_plugin_path = kwiver::vital::get_executable_path() +
    "/../lib/kwiver/modules";
    kwiver::vital::plugin_manager::instance().add_search_path(rel_plugin_path);
    kwiver::vital::plugin_manager::instance().load_all_plugins();
  }

  void initialize()
  {
    // Create algorithm from configuration
    initialize_cameras_landmarks::set_nested_algo_configuration( "initializer",
                                                                 config,
                                                                 algorithm );
  }

  void clear_ptrs()
  {
    camera_map_ptr = nullptr;
    landmark_map_ptr = nullptr;
    feature_track_set_ptr = nullptr;
    sfm_constraint_ptr = nullptr;
  }

  void load_tracks( )
  {
    kwiver::vital::path_t in_tracks_path =
      config->get_value<kwiver::vital::path_t>("input_tracks_file");
    feature_track_set_ptr =
      kwiver::vital::read_feature_track_file(in_tracks_path);
  }

  void load_sfm_constraint( )
  {
    if(config == nullptr)
    {
      return;
    }

    sfm_constraint_ptr = std::make_shared<sfm_constraints>();
    if(config->has_value("video_source") &&
       config->get_value<std::string>("video_source") != "")
    {
      video_input_sptr video_reader;
      video_input::set_nested_algo_configuration("video_reader", config,
                                                 video_reader);
      video_input::get_nested_algo_configuration("video_reader", config,
                                                 video_reader);

      video_reader->open( config->get_value<std::string>("video_source") );
      if (video_reader->get_implementation_capabilities()
        .has_capability(video_input::HAS_METADATA))
      {
        kwiver::vital::timestamp currentTimestamp;
        kwiver::vital::metadata_map::map_metadata_t map_t;
        while (video_reader->next_frame(currentTimestamp))
        {
          auto frame = currentTimestamp.get_frame();
          auto mdVec = video_reader->frame_metadata();

          if (mdVec.size() > 0)
          {
            map_t.emplace(frame, mdVec);
          }
        }
        kwiver::vital::metadata_map_sptr metadata_map =
        std::make_shared<kwiver::vital::simple_metadata_map>(map_t);
        sfm_constraint_ptr->set_metadata(metadata_map);
      }
    }
  }

  bool write_cameras()
  {
    std::string output_cameras_director =
    config->get_value<std::string>("output_cameras_director");
    std::string camera_prefix = config->get_value<std::string>("camera_prefix");

    for( auto iter: camera_map_ptr->cameras())
    {
      int fn = iter.first;
      camera_sptr cam = iter.second;
      std::string out_fname = output_cameras_director + "/" + camera_prefix +
      std::to_string(fn) + ".krtd";
      kwiver::vital::path_t out_path(out_fname);
      auto cam_ptr = std::dynamic_pointer_cast<camera_perspective>(cam);
      write_krtd_file( *cam_ptr, out_path );
    }

    return true;
  }

  bool write_landmarks()
  {
    kwiver::vital::path_t out_landmarks_path =
      config->get_value<kwiver::vital::path_t>("output_landmarks_filename");

    // verify that we can open the output file for writing
    // so that we don't find a problem only after spending
    // hours of computation time.
    std::ofstream ofs(out_landmarks_path.c_str());
    if (!ofs)
    {
      LOG_ERROR(main_logger, "Could not open track file for writing: \""
                << out_landmarks_path << "\"");
      return false;
    }
    ofs.close();

    write_ply_file( landmark_map_ptr, out_landmarks_path );
    return true;
  }

  void run_algorithm()
  {
    // If camera_map_ptr is Null the initialize algorithm will create all
    // cameras.  If not Null it will only create cameras if they are in the map
    // but Null.  So we need to add placeholders for missing cameras to the map
    if (camera_map_ptr)
    {
      using kwiver::vital::frame_id_t;
      using kwiver::vital::camera_map;
      std::set<frame_id_t> frame_ids = feature_track_set_ptr->all_frame_ids();
      num_frames = frame_ids.size();
      camera_map::map_camera_t all_cams = camera_map_ptr->cameras();

      for (auto const& id : frame_ids)
      {
        if (all_cams.find(id) == all_cams.end())
        {
          all_cams[id] = kwiver::vital::camera_sptr();
        }
      }
      camera_map_ptr = std::make_shared<kwiver::vital::simple_camera_map>(all_cams);
    }

    // If landmark_map_ptr is Null the initialize algorithm will create all
    // landmarks.  If not Null it will only create landmarks if they are in the
    // map but Null.  So we need to add placeholders for missing landmarks to
    // the map.
    if (landmark_map_ptr)
    {
      using kwiver::vital::track_id_t;
      using kwiver::vital::landmark_map;
      std::set<track_id_t> track_ids = feature_track_set_ptr->all_track_ids();
      landmark_map::map_landmark_t all_lms = landmark_map_ptr->landmarks();

      for (auto const& id : track_ids)
      {
        if (all_lms.find(id) == all_lms.end())
        {
          all_lms[id] = kwiver::vital::landmark_sptr();
        }
      }
      landmark_map_ptr =
        std::make_shared<kwiver::vital::simple_landmark_map>(all_lms);
    }

    algorithm->initialize(camera_map_ptr, landmark_map_ptr,
                          feature_track_set_ptr, sfm_constraint_ptr);
  }
};

// ----------------------------------------------------------------------------
int
init_cameras_landmarks::
run()
{
  try
  {
    d->register_algorithms();

    switch(d->process_command_line(command_args()))
    {
      case priv::HELP:
        std::cout << m_cmd_options->help();
        return EXIT_SUCCESS;
      case priv::WRITE:
        return EXIT_SUCCESS;
      case priv::FAIL:
        return EXIT_FAILURE;
      case priv::SUCCESS:
        ;
    }

    if ( d->config == nullptr )
    {
      return EXIT_FAILURE;
    }

    if(d->algorithm == nullptr)
    {
      d->initialize();
    }

    if(d->feature_track_set_ptr == nullptr)
    {
      d->load_tracks();
      if(d->feature_track_set_ptr == nullptr)
      {
        LOG_ERROR(main_logger, "There are no feature tracks.");
        return EXIT_FAILURE;
      }
    }

    if(d->sfm_constraint_ptr == nullptr)
    {
      d->load_sfm_constraint();
    }

    d->run_algorithm();

    if(!d->write_cameras())
    {
      return EXIT_FAILURE;
    }

    if(!d->write_landmarks())
    {
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }
  catch (std::exception const& e)
  {
    LOG_ERROR(main_logger, "Exception caught: " << e.what());

    return EXIT_FAILURE;
  }
  catch (...)
  {
    LOG_ERROR(main_logger, "Unknown exception caught");

    return EXIT_FAILURE;
  }
} // run

// ----------------------------------------------------------------------------
void
init_cameras_landmarks::
add_command_options()
{
  m_cmd_options->custom_help( wrap_text( "[options]\n" ) );

  m_cmd_options->add_options()
  ( "h,help",     "Display applet usage" )
  ( "c,config",   "Configuration file for tool", cxxopts::value<std::string>() )
  ( "o,output-config",
    "Output a configuration. This may be seeded with a "
    "configuration file from -c/--config.",
    cxxopts::value<std::string>() )
  ;
}

// ============================================================================
init_cameras_landmarks::
init_cameras_landmarks()
 : d(new priv)
{ }

} } } // end namespace
