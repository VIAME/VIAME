// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "init_cameras_landmarks.h"

#include <arrows/mvg/sfm_utils.h>
#include <arrows/mvg/transform.h>

#include <kwiversys/SystemTools.hxx>

#include <vital/algo/initialize_cameras_landmarks.h>
#include <vital/algo/video_input.h>
#include <vital/applets/applet_config.h>
#include <vital/applets/config_validation.h>
#include <vital/config/config_block_io.h>
#include <vital/config/config_block.h>
#include <vital/config/config_parser.h>
#include <vital/io/camera_from_metadata.h>
#include <vital/io/camera_io.h>
#include <vital/io/landmark_map_io.h>
#include <vital/io/metadata_io.h>
#include <vital/io/track_set_io.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/util/get_paths.h>

#include <fstream>
#include <iostream>

namespace kwiver {
namespace arrows {
namespace mvg {

namespace kv = kwiver::vital;
using kv::feature_track_set_sptr;
using kv::algo::initialize_cameras_landmarks;
using kv::algo::initialize_cameras_landmarks_sptr;
using kv::algo::video_input;
using kv::algo::video_input_sptr;
using kv::camera_map_sptr;
using kv::camera_perspective;
using kv::camera_sptr;
using kv::landmark_map_sptr;
using kv::sfm_constraints;
using kv::sfm_constraints_sptr;

namespace {

typedef kwiversys::SystemTools ST;

kv::logger_handle_t main_logger( kv::get_logger( "init_cameras_landmarks" ) );

// ------------------------------------------------------------------
bool check_config(kv::config_block_sptr config)
{
  using namespace kwiver::tools;
  bool config_valid = true;

#define KWIVER_CONFIG_FAIL(msg) \
  LOG_ERROR(main_logger, "Config Check Fail: " << msg); \
  config_valid = false

  config_valid =
    validate_required_input_file("input_tracks_file", *config, main_logger)
    && config_valid;

  config_valid =
    validate_optional_input_file("video_source", *config, main_logger)
    && config_valid;

  config_valid =
    validate_required_output_dir("output_cameras_directory", *config, main_logger)
    && config_valid;

  config_valid =
    validate_required_output_file("output_landmarks_filename", *config, main_logger)
    && config_valid;

  config_valid =
    validate_optional_output_file("geo_origin_filename", *config, main_logger)
    && config_valid;

  if (!video_input::check_nested_algo_configuration("video_reader", config))
  {
    KWIVER_CONFIG_FAIL("video_reader configuration check failed");
  }

  if (!initialize_cameras_landmarks::check_nested_algo_configuration("initializer", config))
  {
    KWIVER_CONFIG_FAIL("initializer configuration check failed");
  }

#undef KWIVER_CONFIG_FAIL

  return config_valid;
}
} // end namespace

class init_cameras_landmarks::priv
{
public:
  priv(init_cameras_landmarks* parent) : p(parent) {}

  init_cameras_landmarks* p = nullptr;
  camera_map_sptr camera_map_ptr;
  landmark_map_sptr landmark_map_ptr;
  feature_track_set_sptr feature_track_set_ptr;
  sfm_constraints_sptr sfm_constraint_ptr;
  initialize_cameras_landmarks_sptr algorithm;
  kv::config_block_sptr config;
  size_t num_frames = 0;
  kv::path_t  video_file;
  kv::path_t  tracks_file;
  kv::path_t  camera_directory = "results/krtd";
  kv::path_t  landmarks_file = "results/landmarks.ply";
  kv::path_t  geo_origin_file = "results/geo_origin.txt";

  enum commandline_mode {SUCCESS, HELP, WRITE, FAIL};

  commandline_mode process_command_line(cxxopts::ParseResult& cmd_args)
  {
    using kwiver::tools::load_default_video_input_config;
    static std::string opt_config;
    static std::string opt_out_config;

    if ( cmd_args["help"].as<bool>() )
    {
      return HELP;
    }
    if ( cmd_args.count("config") > 0 )
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
      config->merge_config(kv::read_config_file(opt_config));
    }

    if ( cmd_args.count("tracks") > 0 )
    {
      tracks_file = cmd_args["tracks"].as<std::string>();
      config->set_value("input_tracks_file", tracks_file);

    }
    if ( cmd_args.count("video") > 0 )
    {
      video_file = cmd_args["video"].as<std::string>();
      config->set_value("video_source", video_file);
      // choose video or image list reader based on file extension
      config->subblock_view("video_reader")->merge_config(
        load_default_video_input_config(video_file));
    }
    if ( cmd_args.count("carmera") > 0 )
    {
      camera_directory = cmd_args["carmera"].as<std::string>();
      config->set_value("output_cameras_directory", camera_directory);
    }
    if ( cmd_args.count("landmarks") > 0 )
    {
      landmarks_file = cmd_args["landmarks"].as<std::string>();
      config->set_value("output_landmarks_filename", landmarks_file);
    }
    if (cmd_args.count("geo-origin") > 0)
    {
      geo_origin_file = cmd_args["geo-origin"].as<std::string>();
      config->set_value("geo_origin_filename", geo_origin_file);
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

  kv::config_block_sptr default_config()
  {
    using kwiver::tools::load_default_video_input_config;
    typedef kwiver::tools::kwiver_applet kvt;
    auto config = kvt::find_configuration("applets/init_cameras_landmarks.conf");

    // choose video or image list reader based on file extension
    config->subblock_view("video_reader")->merge_config(
      load_default_video_input_config(video_file));

    config->set_value("video_source", video_file,
      "(optional) Path to an input file to be opened as a video. "
      "This could be either a video file or a text file "
      "containing new-line separated paths to sequential "
      "image files. In this tool, video is only used to extract "
      "metadata such as geospatial tags.");

    config->set_value("input_tracks_file", tracks_file,
      "Path to a file to read input tracks from.");

    config->set_value("output_cameras_directory", camera_directory,
      "Directory to write cameras to.");

    config->set_value("output_landmarks_filename", landmarks_file,
      "Path to a file to write output landmarks to. If this "
      "file exists, it will be overwritten.");

    config->set_value("geo_origin_filename", geo_origin_file,
      "Path to a file to write the geographic origin. "
      "This file is only written if the geospatial metadata is "
      "provided as input (e.g. in the input video).  If this "
      "file exists, it will be overwritten.");

    initialize_cameras_landmarks::get_nested_algo_configuration(
      "initializer", config, nullptr);
    video_input::get_nested_algo_configuration(
      "video_reader", config, nullptr);
    return config;
  }

  void initialize()
  {
    // Create algorithm from configuration
    initialize_cameras_landmarks::set_nested_algo_configuration(
      "initializer", config, algorithm );
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
    if(config == nullptr)
    {
      return;
    }
    tracks_file =
      config->get_value<kv::path_t>("input_tracks_file");
    feature_track_set_ptr =
      kv::read_feature_track_file(tracks_file);
  }

  void load_sfm_constraint( )
  {
    if(config == nullptr)
    {
      return;
    }

    sfm_constraint_ptr = std::make_shared<sfm_constraints>();
    kv::image_container_sptr first_frame;
    if(config->has_value("video_source") &&
       config->get_value<std::string>("video_source") != "")
    {
      video_input_sptr video_reader;
      video_file = config->get_value<std::string>("video_source");
      video_input::set_nested_algo_configuration(
        "video_reader", config, video_reader);
      video_reader->open( video_file );
      if (video_reader->get_implementation_capabilities()
        .has_capability(video_input::HAS_METADATA))
      {
        sfm_constraint_ptr->set_metadata(video_reader->metadata_map());
        kv::timestamp ts;
        video_reader->next_frame( ts );
        first_frame = video_reader->frame_image();
      }
      else
      {
        return;
      }
    }
    else
    {
      return;
    }

    using kv::simple_camera_intrinsics;
    using kv::simple_camera_perspective;
    using kv::frame_id_t;
    using kv::metadata_sptr;
    using kv::intrinsics_from_metadata;
    using kv::local_geo_cs;

    #define GET_K_CONFIG(type, name) \
    config->get_value<type>(bc + #name, K_def.name())

    simple_camera_intrinsics K_def;
    const std::string bc = "video_reader:base_camera:";
    auto K = std::make_shared<simple_camera_intrinsics>(
      GET_K_CONFIG(double, focal_length),
      GET_K_CONFIG(kv::vector_2d, principal_point),
      GET_K_CONFIG(double, aspect_ratio),
      GET_K_CONFIG(double, skew));

    auto base_camera = simple_camera_perspective();
    base_camera.set_intrinsics(K);
    auto md = sfm_constraint_ptr->get_metadata()->metadata();
    if (!md.empty())
    {
      std::map<frame_id_t, metadata_sptr> md_map;

      for (auto const& md_iter : md)
      {
        // TODO: just using first element of metadata vector for now
        md_map[md_iter.first] = md_iter.second[0];
      }

      bool init_cams_with_metadata =
        config->get_value<bool>("initialize_cameras_with_metadata", true);

      if (init_cams_with_metadata)
      {
        auto im = first_frame;
        K->set_image_width(static_cast<unsigned>(im->width()));
        K->set_image_height(static_cast<unsigned>(im->height()));
        base_camera.set_intrinsics(K);

        bool init_intrinsics_with_metadata =
        config->get_value<bool>("initialize_intrinsics_with_metadata", true);
        if (init_intrinsics_with_metadata)
        {
          // find the first metadata that gives valid intrinsics
          // and put this in baseCamera as a backup for when
          // a particular metadata packet is missing data
          for (auto mdp : md_map)
          {
            auto md_K =
              intrinsics_from_metadata(*mdp.second,
                                       static_cast<unsigned>(im->width()),
                                       static_cast<unsigned>(im->height()));
            if (md_K != nullptr)
            {
              base_camera.set_intrinsics(md_K);
              break;
            }
          }
        }

        local_geo_cs lgcs = sfm_constraint_ptr->get_local_geo_cs();
        kv::camera_map::map_camera_t cam_map =
          initialize_cameras_with_metadata(md_map, base_camera, lgcs,
                                           init_intrinsics_with_metadata);
        camera_map_ptr =
          std::make_shared<kv::simple_camera_map>(cam_map);

        sfm_constraint_ptr->set_local_geo_cs(lgcs);
      }
    }
  }

  bool write_cameras()
  {
    std::string output_cameras_directory =
    config->get_value<std::string>("output_cameras_directory");

    for( auto iter: camera_map_ptr->cameras())
    {
      int fn = iter.first;
      camera_sptr cam = iter.second;
      std::string out_fname =
        output_cameras_directory + "/" + get_filename(fn) + ".krtd";
      kv::path_t out_path(out_fname);
      auto cam_ptr = std::dynamic_pointer_cast<camera_perspective>(cam);
      write_krtd_file( *cam_ptr, out_path );
    }

    return true;
  }

  bool write_landmarks()
  {
    kv::path_t out_landmarks_path =
      config->get_value<kv::path_t>("output_landmarks_filename");

    write_ply_file( landmark_map_ptr, out_landmarks_path );
    return true;
  }

  bool write_geo_origin()
  {
    auto lgcs = sfm_constraint_ptr->get_local_geo_cs();
    if (!lgcs.origin().is_empty())
    {
      kv::write_local_geo_cs_to_file(lgcs, geo_origin_file);
      return true;
    }
    return false;
  }

  void run_algorithm()
  {
    // If camera_map_ptr is Null the initialize algorithm will create all
    // cameras.  If not Null it will only create cameras if they are in the map
    // but Null.  So we need to add placeholders for missing cameras to the map
    if (camera_map_ptr)
    {
      using kv::frame_id_t;
      using kv::camera_map;
      std::set<frame_id_t> frame_ids = feature_track_set_ptr->all_frame_ids();
      num_frames = frame_ids.size();
      camera_map::map_camera_t all_cams = camera_map_ptr->cameras();

      for (auto const& id : frame_ids)
      {
        if (all_cams.find(id) == all_cams.end())
        {
          all_cams[id] = kv::camera_sptr();
        }
      }
      camera_map_ptr = std::make_shared<kv::simple_camera_map>(all_cams);
    }

    // If landmark_map_ptr is Null the initialize algorithm will create all
    // landmarks.  If not Null it will only create landmarks if they are in the
    // map but Null.  So we need to add placeholders for missing landmarks to
    // the map.
    if (landmark_map_ptr)
    {
      using kv::track_id_t;
      using kv::landmark_map;
      std::set<track_id_t> track_ids = feature_track_set_ptr->all_track_ids();
      landmark_map::map_landmark_t all_lms = landmark_map_ptr->landmarks();

      for (auto const& id : track_ids)
      {
        if (all_lms.find(id) == all_lms.end())
        {
          all_lms[id] = kv::landmark_sptr();
        }
      }
      landmark_map_ptr =
        std::make_shared<kv::simple_landmark_map>(all_lms);
    }

    algorithm->initialize(camera_map_ptr, landmark_map_ptr,
                          feature_track_set_ptr, sfm_constraint_ptr);
  }

  std::string get_filename( kv::frame_id_t frame_id )
  {

    if (sfm_constraint_ptr && sfm_constraint_ptr->get_metadata())
    {
      auto videoMetadataMap = sfm_constraint_ptr->get_metadata();
      auto mdv = videoMetadataMap->get_vector(frame_id);
      if (!mdv.empty())
      {
        return basename_from_metadata(mdv, frame_id);
      }
    }
    auto dummy_md = std::make_shared<kv::metadata>();
    dummy_md->add<kv::VITAL_META_VIDEO_URI>(std::string(video_file));
    return basename_from_metadata(dummy_md, frame_id);
  }

  void recenter_to_landmarks()
  {
    kv::vector_3d offset = landmarks_ground_center(*landmark_map_ptr);

    auto lgcs = sfm_constraint_ptr->get_local_geo_cs();
    if (!lgcs.origin().is_empty())
    {
      kwiver::vital::vector_3d new_origin = lgcs.origin().location() + offset;
      lgcs.set_origin(kv::geo_point(new_origin, lgcs.origin().crs()));
      sfm_constraint_ptr->set_local_geo_cs(lgcs);
    }

    translate_inplace(*landmark_map_ptr, -offset);
    translate_inplace(*camera_map_ptr, -offset);
  }
};

// ----------------------------------------------------------------------------
int
init_cameras_landmarks::
run()
{
  try
  {
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

    d->recenter_to_landmarks();

    if(!d->write_cameras())
    {
      return EXIT_FAILURE;
    }

    if(!d->write_landmarks())
    {
      return EXIT_FAILURE;
    }

    if (d->write_geo_origin())
    {
      LOG_INFO(main_logger, "Saved geo-origin to " << d->geo_origin_file);
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
  ( "v,video", "Input video", cxxopts::value<std::string>() )
  ( "t,tracks", "Input tracks", cxxopts::value<std::string>() )
  ( "k,camera", "Output directory for cameras", cxxopts::value<std::string>() )
  ( "l,landmarks", "Output landmarks file", cxxopts::value<std::string>() )
  ( "g,geo-origin", "Output geographic origin file", cxxopts::value<std::string>())
  ;

  //If we want to remove tracks reading from the config, we should then add this
  //m_cmd_options->parse_positional("tracks");
}

// ============================================================================
init_cameras_landmarks::
init_cameras_landmarks()
 : d(new priv(this))
{ }

init_cameras_landmarks::
~init_cameras_landmarks() = default;

} } } // end namespace
