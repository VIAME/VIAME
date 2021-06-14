// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track_features.h"

#include <vital/algo/track_features.h>
#include <vital/algo/compute_ref_homography.h>
#include <vital/algo/video_input.h>
#include <vital/applets/applet_config.h>
#include <vital/applets/config_validation.h>
#include <vital/io/track_set_io.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/config/config_block_io.h>
#include <vital/config/config_block.h>
#include <vital/config/config_parser.h>
#include <vital/util/get_paths.h>
#include <vital/util/transform_image.h>

#include <kwiversys/SystemTools.hxx>
#include <kwiversys/CommandLineArguments.hxx>

#include <arrows/core/colorize.h>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace kwiver {
namespace arrows {
namespace mvg {

namespace kv = ::kwiver::vital;
namespace kva = ::kwiver::vital::algo;

namespace {

typedef kwiversys::SystemTools ST;
typedef kwiversys::CommandLineArguments argT;

kv::logger_handle_t main_logger( kv::get_logger( "track_features_applet" ) );

// ------------------------------------------------------------------
bool check_config(kv::config_block_sptr config)
{
  using namespace kwiver::tools;
  bool config_valid = true;

#define KWIVER_CONFIG_FAIL(msg) \
  LOG_ERROR(main_logger, "Config Check Fail: " << msg); \
  config_valid = false

  config_valid =
    validate_required_input_file("video_source", *config, main_logger)
    && config_valid;

  config_valid =
    validate_optional_input_file("mask_source", *config, main_logger)
    && config_valid;

  config_valid =
    validate_required_output_file("output_tracks_file", *config, main_logger)
    && config_valid;

  config_valid =
    validate_optional_output_file("output_homography_file", *config, main_logger)
    && config_valid;

  if (!kv::algo::video_input::check_nested_algo_configuration("video_reader", config))
  {
    KWIVER_CONFIG_FAIL("video_reader configuration check failed");
  }
  if (!kv::algo::video_input::check_nested_algo_configuration("mask_reader", config))
  {
    KWIVER_CONFIG_FAIL("mask_reader configuration check failed");
  }

  if (!kv::algo::track_features::check_nested_algo_configuration("feature_tracker", config))
  {
    KWIVER_CONFIG_FAIL("feature_tracker configuration check failed");
  }

#undef KWIVER_CONFIG_FAIL

  return config_valid;
}

// uniformly sample max_num items from in_data and add to out_data
template <typename T>
void uniform_subsample(std::vector<T> const& in_data,
                       std::vector<T>& out_data, size_t max_num)
{
  const size_t data_size = in_data.size();
  if (data_size <= max_num)
  {
    // append all the data
    out_data.insert(out_data.end(), in_data.begin(), in_data.end());
    return;
  }

  // select max_num distributed throughout the vector
  for (unsigned i = 0; i < max_num; ++i)
  {
    size_t idx = (i * data_size) / max_num;
    out_data.push_back(in_data[idx]);
  }
}

// select which frames to use for tracking
std::vector<kv::frame_id_t>
select_frames(std::vector<kv::frame_id_t> const& valid_frames,
              std::vector<kv::frame_id_t> const& camera_frames,
              size_t max_frames)
{
  std::vector<kv::frame_id_t> selected_frames;
  uniform_subsample(valid_frames, selected_frames, max_frames);

  if (valid_frames.size() > selected_frames.size() &&
      !camera_frames.empty())
  {
    auto c_itr = camera_frames.begin();
    auto s_itr = selected_frames.begin();

    // Step through each selected frame and if there is no camera data
    // on the frame look at the frames between this selected frame and
    // its neighbor selected frames and see if one of those frames has
    // a camera.  If so, pick the frame with the camera data instead.
    // This retains roughly uniformily distributed frames but gives
    // priority to frames with camera data when nearby.
    for (; s_itr != selected_frames.end(); ++s_itr)
    {
      // Find the next camera frame that is equal or greater.
      // This is done such that *(c_itr - 1) < *s_itr and *c_itr >= *s_itr
      for (; c_itr != camera_frames.end() && *c_itr < *s_itr; ++c_itr);
      if (c_itr != camera_frames.end() && *c_itr == *s_itr)
      {
        // This selected frame already has a camera, so move on
        continue;
      }
      kv::frame_id_t new_s = -1;
      kv::frame_id_t diff =
        std::numeric_limits<kv::frame_id_t>::max();
      // Check the previous selected frame if not at the start
      if (s_itr != selected_frames.begin() &&
          c_itr != camera_frames.begin() )
      {
        // Set a search limit half way to the previous keyframe
        auto prev_lim = (*s_itr + *(s_itr - 1) + 1) / 2;
        if (prev_lim <= *(c_itr - 1))
        {
          new_s = *(c_itr - 1);
          diff = (*s_itr - new_s);
        }
      }
      // Check the next selected frame if not at the end
      if ((s_itr+1) != selected_frames.end() &&
          c_itr != camera_frames.end() )
      {
        // Set a search limit half way to the next keyframe
        auto next_lim = (*s_itr + *(s_itr + 1)) / 2;
        // Use this camera frame only if within the limit and
        // closer than the camera found above
        if (next_lim >= *c_itr && (*c_itr - *s_itr) < diff )
        {
          new_s = *c_itr;
          diff = *c_itr - *s_itr;
        }
      }
      // Update the selection of a new selection was found
      if (new_s >=0)
      {
        *s_itr = new_s;
      }
    }
  }

  return selected_frames;
}

} // end namespace

class track_features::priv
{
public:
  priv(track_features* parent) : p(parent) {}

  track_features* p;
  kva::video_input_sptr video_reader = nullptr;
  kva::video_input_sptr mask_reader = nullptr;
  kva::track_features_sptr feature_tracker = nullptr;
  kva::compute_ref_homography_sptr out_homog_generator = nullptr;
  kv::config_block_sptr config = nullptr;
  std::string video_file;
  std::string mask_file;
  std::string homography_file;
  std::string track_file = "results/tracks.txt";

  enum commandline_mode { SUCCESS, HELP, WRITE, FAIL };

  commandline_mode process_command_line(cxxopts::ParseResult& cmd_args)
  {
    using kwiver::tools::load_default_video_input_config;
    std::string opt_config;
    std::string opt_out_config;

    if (cmd_args["help"].as<bool>())
    {
      return HELP;
    }
    if (cmd_args.count("config"))
    {
      opt_config = cmd_args["config"].as<std::string>();
    }
    if (cmd_args.count("output-config") > 0)
    {
      opt_out_config = cmd_args["output-config"].as<std::string>();
    }

    // Set up top level configuration w/ defaults where applicable.
    config = default_config();

    // If -c/--config given, read in confg file, merge in with default just
    // generated
    if (!opt_config.empty())
    {
      config->merge_config(kv::read_config_file(opt_config));
    }

    if (cmd_args.count("video-file"))
    {
      video_file = cmd_args["video-file"].as<std::string>();
      config->set_value("video_source", video_file);
      // choose video or image list reader based on file extension
      config->subblock_view("video_reader")->merge_config(
        load_default_video_input_config(video_file));
    }

    if (cmd_args.count("mask-file"))
    {
      mask_file = cmd_args["mask-file"].as<std::string>();
      config->set_value("mask_source", mask_file);
      // choose video or image list reader for masks based on file extension
      config->subblock_view("mask_reader")->merge_config(
        load_default_video_input_config(mask_file));
    }

    if (cmd_args.count("homography-file"))
    {
      homography_file = cmd_args["homography-file"].as<std::string>();
      config->set_value("output_homography_file", homography_file);
    }

    if (cmd_args.count("track-file"))
    {
      track_file = cmd_args["track-file"].as<std::string>();
      config->set_value("output_tracks_file", track_file);
    }

    bool valid_config = check_config(config);

    if (!opt_out_config.empty())
    {
      write_config_file(config, opt_out_config);
      if (valid_config)
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
    else if (!valid_config)
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
    using kwiver::tools::load_default_video_input_config;
    typedef kwiver::tools::kwiver_applet kvt;
    auto config = kvt::find_configuration("applets/track_features.conf");

    // choose video or image list reader based on file extension
    config->subblock_view("video_reader")->merge_config(
      load_default_video_input_config(video_file));

    // choose video or image list reader for masks based on file extension
    config->subblock_view("mask_reader")->merge_config(
      load_default_video_input_config(mask_file));

    config->set_value("video_source", video_file,
      "Path to an input file to be opened as a video. "
      "This could be either a video file or a text file "
      "containing new-line separated paths to sequential "
      "image files.");

    config->set_value("output_tracks_file", track_file,
      "Path to a file to write output tracks to. If this "
      "file exists, it will be overwritten.");

    config->set_value("mask_source", mask_file,
      "Optional path to an mask input file to be opened "
      "as a video. "
      "This could be either a video file or a text file "
      "containing new-line separated paths to sequential "
      "image files. "
      "This list should be "
      "parallel in association to frames provided by the "
      "``video_source`` video. Mask images must be the same size "
      "as the image they are associated with.\n"
      "\n"
      "Leave this blank if no image masking is desired.");

    config->set_value("output_homography_file", homography_file,
      "Optional path to a file to write source-to-reference "
      "homographies for each frame. Leave blank to disable this "
      "output. The output_homography_generator algorithm type "
      "only needs to be set if this is set.");

    config->set_value("invert_masks", false,
      "If true, all mask images will be inverted after loading. "
      "This is useful if mask images read in use positive "
      "values to indicated masked areas instead of non-masked "
      "areas.");

    config->set_value("use_video_metadata", true,
      "If true, extract metadata from the video");

    config->set_value("expect_multichannel_masks", false,
      "A majority of the time, mask images are a single channel, "
      "however it is feasibly possible that certain "
      "implementations may use multi-channel masks. If this is "
      "true we will expect multiple-channel mask images, "
      "warning when a single-channel mask is provided. If this "
      "is false we error upon seeing a multi-channel mask "
      "image.");

    kva::video_input::get_nested_algo_configuration("video_reader", config,
      kva::video_input_sptr());
    kva::video_input::get_nested_algo_configuration("mask_reader", config,
      kva::video_input_sptr());
    kva::track_features::get_nested_algo_configuration("feature_tracker", config,
      kva::track_features_sptr());
    kva::compute_ref_homography::get_nested_algo_configuration("output_homography_generator",
      config, kva::compute_ref_homography_sptr());

    return config;
  }

  void initialize()
  {
    kva::video_input::set_nested_algo_configuration(
      "video_reader", config, video_reader);
    kva::video_input::set_nested_algo_configuration(
      "mask_reader", config, mask_reader);
    kva::track_features::set_nested_algo_configuration(
      "feature_tracker", config, feature_tracker);
    kva::compute_ref_homography::set_nested_algo_configuration(
      "output_homography_generator", config, out_homog_generator);
  }

  bool run_algorithm()
  {
    // Attempt opening input and output files.
    //  - filepath validity checked above
    video_file = config->get_value<std::string>("video_source");
    mask_file = config->get_value<std::string>("mask_source");
    track_file = config->get_value<std::string>("output_tracks_file");

    auto const invert_masks = config->get_value<bool>("invert_masks");
    auto const use_video_metadata = config->get_value<bool>("use_video_metadata");
    auto const expect_multichannel_masks = config->get_value<bool>("expect_multichannel_masks");
    auto const max_frames = config->get_value<unsigned int>("feature_tracker:max_frames");

    bool hasMask = !mask_file.empty();
    LOG_INFO(main_logger, "Reading Video");
    video_reader->open(video_file);
    if (hasMask)
    {
      mask_reader->open(mask_file);
    }

    kv::metadata_map_sptr md_map;
    std::vector<kv::frame_id_t> valid_frames;
    std::vector<kv::frame_id_t> camera_frames;
    if (use_video_metadata && (md_map = video_reader->metadata_map()) &&
        md_map->size() > 0)
    {
      auto fs = md_map->frames();
      valid_frames = std::vector<kv::frame_id_t>(fs.begin(), fs.end());
      camera_frames = valid_frames;
    }
    else
    {
      auto const num_frames = video_reader->num_frames();
      auto const max_frame = static_cast<kv::frame_id_t>(num_frames);

      valid_frames.reserve(num_frames);
      for (kv::frame_id_t f = 1; f <= max_frame; ++f)
      {
        valid_frames.push_back(f);
      }
    }

    std::vector<kv::frame_id_t> selected_frames =
      select_frames(valid_frames, camera_frames, max_frames);

    kv::timestamp currentTimestamp;

    // verify that we can open the output file for writing
    // so that we don't find a problem only after spending
    // hours of computation time.
    std::ofstream ofs(track_file.c_str());
    if (!ofs)
    {
      LOG_ERROR(main_logger, "Could not open track file for writing: \""
                             << track_file << "\"");
      return false;
    }
    ofs.close();

    // Create the output homography file stream if specified
    // Validity of file path checked during configuration file validity check.
    std::ofstream homog_ofs;
    if (config->has_value("output_homography_file") &&
        config->get_value<std::string>("output_homography_file") != "")
    {
      kv::path_t homog_fp = config->get_value<kv::path_t>("output_homography_file");
      homog_ofs.open(homog_fp.c_str());
      if (!homog_ofs)
      {
        LOG_ERROR(main_logger, "Could not open homography file for writing: "
                               << homog_fp);
        return false;
      }
    }

    // Track features on each frame sequentially
    kv::feature_track_set_sptr tracks;
    for (auto target_frame : selected_frames)
    {
      bool valid = true;
      kv::frame_id_t frame = 0;

      // step to find the next target frame
      do
      {
        valid = video_reader->next_frame(currentTimestamp);
        if (hasMask)
        {
          kv::timestamp dummyTimestamp;
          valid = valid && mask_reader->next_frame(dummyTimestamp);
        }
        frame = currentTimestamp.get_frame();
      } while (valid && frame < target_frame);
      if (!valid)
      {
        break;
      }

      auto const image = video_reader->frame_image();
      kv::image_container_sptr mask;
      if (hasMask)
      {
        mask = mask_reader->frame_image();

        // error out if we are not expecting a multi-channel mask
        if (!expect_multichannel_masks && mask->depth() > 1)
        {
          LOG_ERROR(main_logger, "Encounted multi-channel mask image!");
          return false;
        }
        else if (expect_multichannel_masks && mask->depth() == 1)
        {
          LOG_WARN(main_logger,
            "Expecting multi-channel masks but received one that was "
            "single-channel.");
        }

        if (invert_masks)
        {
          LOG_DEBUG(main_logger, "Inverting mask image pixels");
          kv::image_of<bool> mask_image;
          kv::cast_image(mask->get_image(), mask_image);
          kv::transform_image(mask_image, [](bool b) { return !b; });
          LOG_DEBUG(main_logger, "Inverting mask image pixels -- Done");
          mask = std::make_shared<kv::simple_image_container>(mask_image);
        }

      }
      auto const mdv = video_reader->frame_metadata();
      if (!mdv.empty())
      {
        image->set_metadata(mdv[0]);
      }

      tracks = feature_tracker->track(tracks, frame, image, mask);
      if (tracks)
      {
        tracks = kwiver::arrows::core::extract_feature_colors(tracks, *image, frame);
      }

      // Compute ref homography for current frame with current track set + write to file
      // -> still doesn't take into account a full shotbreak, which would incur a track reset
      if (homog_ofs.is_open())
      {
        LOG_DEBUG(main_logger, "writing homography");
        homog_ofs << *(out_homog_generator->estimate(frame, tracks)) << std::endl;
      }
    }
    video_reader->close();
    if (hasMask)
    {
      mask_reader->close();
    }

    if (homog_ofs.is_open())
    {
      homog_ofs.close();
    }

    // Writing out tracks to file
    kv::write_feature_track_file(tracks, track_file);

    return true;
  }
};

// ----------------------------------------------------------------------------
int
track_features::
run()
{

  try
  {
    switch (d->process_command_line(command_args()))
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

    if (d->config == nullptr)
    {
      return EXIT_FAILURE;
    }

    d->initialize();

    if (!d->run_algorithm())
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
track_features::
add_command_options()
{
  m_cmd_options->custom_help( wrap_text(
    "[options] video-file [track-file]\n"
    "This program tracks feature point through a video or list of images and "
    "produces a track file and optional homography sequence.") );

  m_cmd_options->positional_help("\n  video-file  - name of input video file."
                                 "\n  track-file  - name of output track file "
                                 "(default: " + d->track_file + ")");

  m_cmd_options->add_options()
    ( "h,help",     "Display applet usage" )
    ( "c,config",   "Configuration file for tool", cxxopts::value<std::string>() )
    ( "o,output-config",
      "Output a configuration. This may be seeded with a "
      "configuration file from -c/--config.",
      cxxopts::value<std::string>() )
    ( "g,homography-file",
      "An output homography file containing a sequence of homographies "
      "aligning one frame to another estimated from the tracks.",
      cxxopts::value<std::string>())
    ( "m,mask-file",
      "An input mask video or list of mask images to indicate "
      "which pixels to ignore.",
      cxxopts::value<std::string>())

    // positional parameters
    ("video-file", "Video input file", cxxopts::value<std::string>())
    ("track-file", "Tracks output file", cxxopts::value<std::string>())
    ;

  m_cmd_options->parse_positional({ "video-file", "track-file" });
}

// ============================================================================
track_features::
track_features()
  : d(new priv(this))
{ }

track_features::
~track_features() = default;

} } } // end namespace
