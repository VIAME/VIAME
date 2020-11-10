// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "estimate_depth.h"

#include <arrows/core/depth_utils.h>
#include <arrows/vtk/depth_utils.h>
#include <vital/algo/compute_depth.h>
#include <vital/algo/video_input.h>
#include <vital/applets/applet_config.h>
#include <vital/applets/config_validation.h>
#include <vital/exceptions/io.h>
#include <vital/io/camera_io.h>
#include <vital/io/landmark_map_io.h>
#include <vital/io/metadata_io.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/config/config_block_io.h>
#include <vital/config/config_block.h>
#include <vital/config/config_parser.h>
#include <vital/types/camera_perspective.h>

#include <vtkXMLImageDataWriter.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

namespace kwiver {
namespace arrows {
namespace vtk {

namespace kv = ::kwiver::vital;
namespace kva = ::kwiver::vital::algo;

using kwiver::vital::algo::compute_depth;
using kwiver::vital::algo::video_input;
using kwiver::vital::camera_perspective;
using kwiver::arrows::core::find_similar_cameras_angles;
using kwiver::arrows::core::gather_depth_frames;
using kwiver::arrows::core::compute_robust_ROI;


namespace {

kv::logger_handle_t main_logger( kv::get_logger( "estimate_depth_tool" ) );

// ------------------------------------------------------------------
bool check_config(kv::config_block_sptr config)
{
  using namespace kwiver::tools;
  bool config_valid = true;

#define KWIVER_CONFIG_FAIL(msg) \
  LOG_ERROR(main_logger, "Config Check Fail: " << msg); \
  config_valid = false

  config_valid = validate_required_input_file("video_source", *config, main_logger) && config_valid;

  config_valid = validate_required_input_file("input_landmarks_file", *config, main_logger) && config_valid;

  config_valid = validate_optional_input_file("mask_source", *config, main_logger) && config_valid;

  // This functions as checking for a required input directory, by not allowing
  // it to be created
  config_valid = validate_required_output_dir("input_cameras_directory", *config, main_logger, false) && config_valid;

  config_valid = validate_required_output_dir("output_depths_directory", *config, main_logger) && config_valid;

  if (!kva::video_input::check_nested_algo_configuration("video_reader", config))
  {
    KWIVER_CONFIG_FAIL("video_reader configuration check failed");
  }

  if (!kva::compute_depth::check_nested_algo_configuration("compute_depth", config))
  {
    KWIVER_CONFIG_FAIL("compute_depth configuration check failed");
  }

  if ( config->has_value("mask_source") )
  {
    if (!kva::video_input::check_nested_algo_configuration("mask_reader", config))
    {
      KWIVER_CONFIG_FAIL("mask_reader configuration check failed");
    }
  }

#undef KWIVER_CONFIG_FAIL

  return config_valid;
}
} // end namespace

class estimate_depth::priv
{
public:
  priv()
  : has_mask(false),
    input_cameras_directory("results/krtd"),
    input_landmarks_file("results/landmarks.ply"),
    output_depths_directory("results/depth")
  {}

  bool has_mask;
  kv::path_t input_cameras_directory;
  kv::path_t input_landmarks_file;
  kv::path_t output_depths_directory;

  kv::landmark_map_sptr landmark_map;
  kv::camera_map_sptr camera_map;
  kva::video_input_sptr video_reader;
  kva::video_input_sptr mask_reader;
  kv::metadata_map_sptr metadata_map;
  kv::config_block_sptr config;
  kva::compute_depth_sptr depth_algo;

  kv::path_t video_source;
  kv::path_t mask_source;

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
      config->merge_config(kv::read_config_file(opt_config));
    }

    if ( cmd_args.count("frame") > 0 )
    {
      const int frame = cmd_args["frame"].as<int>();
      config->set_value("frame_index", frame);
    }

    if ( cmd_args.count("video-source") > 0 )
    {
      video_source = cmd_args["video-source"].as<std::string>();
      config->set_value("video_source", video_source);

      config->subblock_view("video_reader")->merge_config(
        load_default_video_input_config(video_source));
    }

    if ( cmd_args.count("input-cameras-dir") > 0 )
    {
      input_cameras_directory = cmd_args["input-cameras-dir"].as<std::string>();
      config->set_value("input_cameras_directory", input_cameras_directory);
    }

    if ( cmd_args.count("input-landmarks-file") > 0 )
    {
      input_landmarks_file = cmd_args["input-landmarks-file"].as<std::string>();
      config->set_value("input_landmarks_file", input_landmarks_file);
    }

    if ( cmd_args.count("mask-source") > 0 )
    {
      has_mask = true;
      mask_source = cmd_args["mask-source"].as<std::string>();
      config->set_value("mask_source", mask_source);

      config->subblock_view("mask_reader")->merge_config(
        load_default_video_input_config(mask_source));
    }

    if ( cmd_args.count("output-depths-dir") > 0 )
    {
      output_depths_directory = cmd_args["output-depths-dir"].as<std::string>();
      config->set_value("output_depths_directory", output_depths_directory);
    }

    bool valid_config = check_config(config);

    if( ! opt_out_config.empty() )
    {
      write_config_file( config, opt_out_config );
      if( valid_config )
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
    else if( !valid_config )
    {
      LOG_ERROR(main_logger, "Configuration not valid.");
      config = nullptr;
      return FAIL;
    }

    return SUCCESS;
  }

  // ------------------------------------------------------------------
  kv::config_block_sptr default_config()
  {
    auto config =
      estimate_depth::find_configuration("applets/estimate_depth.conf");

    // choose video or image list reader based on file extension
    config->subblock_view("video_reader")->merge_config(
      kwiver::tools::load_default_video_input_config(video_source));

    config->subblock_view("mask_reader")->merge_config(
      kwiver::tools::load_default_video_input_config(mask_source));

    config->set_value("video_source", video_source,
      "Path to an input file to be opened as a video. "
      "This could be either a video file or a text file "
      "containing new-line separated paths to sequential "
      "image files.");

    config->set_value("mask_source", mask_source,
      "Path to an input file to be opened as a binary "
      "video, with light regions representing good pixels. "
      "This could be either a video file or a text file "
      "containing new-line separated paths to sequential "
      "image files.");

    config->set_value("input_cameras_directory", input_cameras_directory,
      "Path to a directory to read cameras from.");

    config->set_value("input_landmarks_file", input_landmarks_file,
      "Path to a file to read landmarks from.");

    config->set_value("output_depths_directory", output_depths_directory,
      "Path to a directory to write depth estimations.");

    kva::video_input::get_nested_algo_configuration("video_reader", config,
                                        kva::video_input_sptr());
    kva::video_input::get_nested_algo_configuration("mask_reader", config,
                                        kva::video_input_sptr());
    kva::compute_depth::get_nested_algo_configuration("compute_depth", config,
                                        kva::compute_depth_sptr());
    return config;
  }

  void initialize()
  {
    // Create algorithm from configuration
    compute_depth::set_nested_algo_configuration( "compute_depth",
                                                  config,
                                                  depth_algo );
    video_input::set_nested_algo_configuration( "video_reader",
                                                config,
                                                video_reader );
    video_input::set_nested_algo_configuration( "mask_reader",
                                                config,
                                                mask_reader );
  }

  void load_landmarks( )
  {
    landmark_map = read_ply_file(input_landmarks_file);
  }

  void load_camera_map()
  {
    if ( config == nullptr || video_reader == nullptr )
    {
      return;
    }

    const std::string video_source =
      config->get_value<std::string>("video_source");

    video_reader->open(video_source);

    if ( metadata_map == nullptr )
    {
      // load the metadata map
      metadata_map = video_reader->metadata_map();
    }

    camera_map::map_camera_t cameras;

    for (auto const& frame_metadata : metadata_map->metadata() )
    {
      const kv::metadata_vector mdv = frame_metadata.second;
      const size_t frame_ID = frame_metadata.first;
      const std::string name = basename_from_metadata(mdv, frame_ID);
      try
      {
        cameras[frame_ID] = read_krtd_file( name, input_cameras_directory );
      }
      catch ( const file_not_found_exception& )
      {
        continue;
      }
    }

    if ( cameras.empty() )
    {
      VITAL_THROW( invalid_data, "No krtd files found" );
    }

    camera_map = camera_map_sptr( new simple_camera_map( cameras ) );
  }

  // ----------------------------------------------------------------------------
  bool write_depth(vtkSmartPointer<vtkImageData> depth_image,
                   kv::frame_id_t frame)
  {

    vtkSmartPointer<vtkXMLImageDataWriter> writer =
       vtkSmartPointer<vtkXMLImageDataWriter>::New();

    auto metadata = metadata_map->metadata();

    if ( !depth_image )
    {
      LOG_ERROR(main_logger, "The depth data for frame "
                             << frame <<
                             " was empty");
      return false;
    }

    auto const mdv = metadata.at(frame);
    const std::string basename = basename_from_metadata(mdv, frame);
    std::string output_filename = output_depths_directory + "/"
                                  + basename + ".vti";
    writer->SetFileName(output_filename.c_str());
    writer->SetInputData(depth_image);
    writer->Write();

    return true;
  }

  std::vector<kv::frame_id_t>
  compute_frames_in_range()
  {
    std::vector<kv::frame_id_t> frames_in_range;
    auto const cameras = camera_map->cameras();
    const kv::frame_id_t start_frame = config->get_value<kv::frame_id_t>("batch_depth:first_frame");
    const kv::frame_id_t end_frame = config->get_value<kv::frame_id_t>("batch_depth:end_frame");

    for (auto itr = cameras.begin(); itr != cameras.end(); itr++)
    {
      if (itr->first >= start_frame && (end_frame < 0 || itr->first <= end_frame))
        frames_in_range.push_back(itr->first);
    }
    return frames_in_range;
  }

  vtkSmartPointer<vtkImageData>
  run_single_frame_algorithm(const kv::camera_perspective_map &pcm,
                             int frame_index, const kv::vector_3d &minpt,
                             const kv::vector_3d &maxpt, double height_min,
                             double height_max)
  {
    auto const cameras = camera_map->cameras();
    auto const ref_cam_itr = cameras.find(frame_index);
    if (ref_cam_itr == cameras.end())
    {
      const std::string msg = "No camera available on the selected frame";
      LOG_DEBUG(main_logger, msg);
      throw kv::invalid_value(msg);
    }
    camera_perspective_sptr ref_cam =
      std::dynamic_pointer_cast<camera_perspective>(ref_cam_itr->second);
    if (!ref_cam)
    {
      const std::string msg = "Reference camera is not perspective";
      LOG_DEBUG(main_logger, msg);
      throw kv::invalid_value(msg);
    }

    const double angle_span =
      config->get_value<double>("compute_depth:angle_span", 15.0);
    const int num_support =
      config->get_value<int>("compute_depth:num_support", 20);
    auto similar_cameras =
      find_similar_cameras_angles(*ref_cam, pcm, angle_span, num_support);
    // make sure the reference frame is included
    similar_cameras->insert(frame_index, ref_cam);

    if (similar_cameras->size() < 2)
    {
      const std::string msg = "Not enough cameras with similar viewpoints";
      LOG_DEBUG(main_logger, msg);
      throw kv::invalid_value(msg);
    }

    LOG_DEBUG(main_logger, "found " << similar_cameras->size()
                           << " cameras within " << angle_span << " degrees");
    for (auto const& item : similar_cameras->cameras())
    {
      LOG_DEBUG(main_logger, "   frame " << item.first);
    }

    // collect all the frames
    std::vector<kv::image_container_sptr> frames_out;
    std::vector<kv::camera_perspective_sptr> cameras_out;
    std::vector<kv::image_container_sptr> masks_out;

    // A nullprt implies that we should not use the mask reader. However,
    // the mask reader may be non-null because it was set up in configuration,
    // even if there are not any masks present
    kv::algo::video_input_sptr valid_mask_reader =
      has_mask ? mask_reader : nullptr;

    auto cb = nullptr;
    int ref_frame = gather_depth_frames(*similar_cameras, video_reader,
                                        valid_mask_reader, frame_index,
                                        cameras_out, frames_out, masks_out, cb);
    kv::image_of<unsigned char> ref_mask;
    if (has_mask)
    {
      ref_mask = masks_out[ref_frame]->get_image();
    }

    kv::image_of<uint8_t> ref_img( frames_out[ref_frame]->get_image() );

    LOG_DEBUG(main_logger, "ref frame at index " << ref_frame
                           << " out of " << frames_out.size());

    auto crop = kwiver::arrows::core::project_3d_bounds(minpt, maxpt, *cameras_out.at(ref_frame),
                                                        static_cast<int>( ref_img.width() ),
                                                        static_cast<int>( ref_img.height() ) );

    LOG_DEBUG(main_logger, "Computing Cost Volume");

    kv::image_container_sptr depth_uncertainty;
    auto depth = depth_algo->compute(frames_out, cameras_out,
                                     height_min, height_max,
                                     ref_frame, crop,
                                     depth_uncertainty, masks_out);


    vtkSmartPointer<vtkImageData> image_data;
    if (depth)
    {
      kv::image_of<double> depth_img( depth->get_image() );
      kv::image_of<double> uncertainty( depth_uncertainty->get_image() );
      image_data = depth_to_vtk( depth_img,
                                 ref_img,
                                 crop,
                                 uncertainty,
                                 ref_mask );
    }
    return image_data;
  }

  int run_algorithm()
  {

    kv::camera_perspective_map pcm;
    pcm.set_from_base_camera_map(camera_map->cameras());

    video_reader->open( video_source );

    if (has_mask)
    {
      mask_reader->open(mask_source);
    }

    // Convert landmarks to vector
    std::vector<kv::landmark_sptr> landmarks_out;
    landmarks_out.reserve(landmark_map->size());
    for (auto l : landmark_map->landmarks())
    {
      landmarks_out.push_back(l.second);
    }

    double bounds[6]; // xmin, xmax, ymin, ymax, zmin, zmax
    compute_robust_ROI(landmarks_out, bounds);
    double height_min, height_max;

    kv::vector_3d minpt({bounds[0], bounds[2], bounds[4]});
    kv::vector_3d maxpt({bounds[1], bounds[3], bounds[5]});

    kwiver::arrows::core::height_range_from_3d_bounds(minpt, maxpt, height_min, height_max);

    std::vector<kv::frame_id_t> frames_in_range;
    // Compute for one frame if specified, otherwise compute a linear sampling
    // from all available frames
    if ( config->has_value("frame_index") )
    {
      const int frame = config->get_value<int>("frame_index");
      LOG_INFO(main_logger, "Processing frame " << frame);

      //compute depth
      auto depth = run_single_frame_algorithm(pcm,
                                              frame, minpt,
                                              maxpt, height_min,
                                              height_max);

      if ( !write_depth(depth, frame) )
      {
        // processing was terminated before any result was produced or the
        // result couldn't be saved
        return EXIT_FAILURE;
      }
    }
    else
    {
      frames_in_range = compute_frames_in_range();
      const size_t num_depth_maps = config->get_value<size_t>("batch_depth:num_depth");

      for (size_t c = 0; c < num_depth_maps; ++c)
      {
        size_t curr_frame_index = (c * (frames_in_range.size()-1)) /
                                  (num_depth_maps-1);

        auto fitr = frames_in_range.begin() + curr_frame_index;
        auto active_frame = *fitr;
        LOG_INFO(main_logger, "Processing frame " << active_frame);

        //compute depth
        auto depth = run_single_frame_algorithm(pcm,
                                                active_frame, minpt,
                                                maxpt, height_min,
                                                height_max);

        if ( !write_depth(depth, active_frame) )
        {
          // processing was terminated before any result was produced or the
          // result couldn't be saved
          return EXIT_FAILURE;
        }
      }
    }
    return EXIT_SUCCESS;
  }
};


// ----------------------------------------------------------------------------
int
estimate_depth::
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

    if( d->depth_algo == nullptr || d->video_reader == nullptr )
    {
      d->initialize();
    }

    if( d->landmark_map == nullptr )
    {
      d->load_landmarks();
      if( d->landmark_map == nullptr )
      {
        LOG_ERROR(main_logger, "There are no landmarks.");
        return EXIT_FAILURE;
      }
    }

    if( d->camera_map == nullptr || d->camera_map->size() == 0 )
    {
      d->load_camera_map();
      if( d->camera_map == nullptr || d->camera_map->size() == 0 )
      {
        LOG_ERROR(main_logger, "There are no cameras.");
        return EXIT_FAILURE;
      }
    }

    LOG_INFO(main_logger, "Finished configuring");
    d->run_algorithm();
    LOG_INFO(main_logger, "Finished computing");

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
estimate_depth::
add_command_options()
{
  m_cmd_options->custom_help( wrap_text( "[options]\n" ) );

  m_cmd_options->positional_help(
    "\n  video-source - name of input video file."
    "\n  input-cameras-dir - name of the directory containing the krtd camera files"
    "(default: " + d->input_cameras_directory + ")"
    "\n  output-depths-dir - name of the directory to write depth maps to "
    "(default: " + d->output_depths_directory + ")" );

  m_cmd_options->add_options()
    ( "h,help",     "Display applet usage" )
    ( "c,config",   "Configuration file for tool", cxxopts::value<std::string>() )
    ( "o,output-config",
      "Output a configuration. This may be seeded with a "
      "configuration file from -c/--config.",
      cxxopts::value<std::string>() )
    ( "f,frame",    "The frame number to compute depth for.",
      cxxopts::value<int>() )
    ("l,input-landmarks-file", "3D sparse features", cxxopts::value<std::string>())
    ("m,mask-source", "Masks of unusable regions", cxxopts::value<std::string>())

    // positional parameters
    ("video-source", "Video input file", cxxopts::value<std::string>())
    ("input-cameras-dir", "Camera location data", cxxopts::value<std::string>())
    ("output-depths-dir", "Output directory for depth maps",
     cxxopts::value<std::string>())
    ;

    m_cmd_options->parse_positional({ "video-source", "input-cameras-dir", "output-depths-dir" });
}

// ============================================================================
estimate_depth::
estimate_depth()
  : d(new priv())
{ }

estimate_depth::
~estimate_depth() = default;

} } } // end namespace
