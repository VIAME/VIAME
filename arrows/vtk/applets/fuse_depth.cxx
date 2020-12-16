// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "fuse_depth.h"

#include <arrows/core/depth_utils.h>
#include <arrows/mvg/sfm_utils.h>
#include <arrows/vtk/depth_utils.h>

#include <kwiversys/Directory.hxx>
#include <kwiversys/SystemTools.hxx>

#include <vital/algo/integrate_depth_maps.h>
#include <vital/applets/applet_config.h>
#include <vital/applets/config_validation.h>
#include <vital/config/config_block_io.h>
#include <vital/config/config_block.h>
#include <vital/config/config_parser.h>
#include <vital/io/camera_io.h>
#include <vital/io/landmark_map_io.h>
#include <vital/io/track_set_io.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/bounding_box.h>
#include <vital/util/get_paths.h>

#include <vtkBox.h>
#include <vtkFlyingEdges3D.h>
#include <vtkPLYWriter.h>
#include <vtkOBJWriter.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLPolyDataWriter.h>

#include <fstream>
#include <iostream>

namespace kwiver {
namespace arrows {
namespace vtk {

namespace kv = kwiver::vital;
namespace kva = ::kwiver::vital::algo;

using kv::feature_track_set_sptr;
using kv::algo::integrate_depth_maps;
using kv::algo::integrate_depth_maps_sptr;
using kv::camera_map_sptr;
using kv::camera_perspective;
using kv::camera_sptr;
using kv::landmark_map_sptr;

using kwiver::arrows::core::compute_robust_ROI;
using kwiver::arrows::mvg::crop_camera;


namespace {
static char const* const CAMERA_EXTENSION = ".krtd";
static char const* const DEPTH_EXTENSION = ".vti";

typedef kwiversys::SystemTools ST;

kv::logger_handle_t main_logger( kv::get_logger( "fuse_depth" ) );


// ------------------------------------------------------------------
bool check_config(kv::config_block_sptr config)
{
  using namespace kwiver::tools;
  bool config_valid = true;

#define KWIVER_CONFIG_FAIL(msg) \
  LOG_ERROR(main_logger, "Config Check Fail: " << msg); \
  config_valid = false

  config_valid =
    validate_required_input_file("input_landmarks_file", *config, main_logger)
    && config_valid;

  config_valid =
    validate_required_input_dir("input_cameras_directory", *config,
                                 main_logger) && config_valid;

  config_valid =
    validate_required_input_dir("input_depths_directory", *config,
                                 main_logger) && config_valid;

  config_valid =
    validate_optional_output_file("output_volume_file", *config,
                                   main_logger) && config_valid;

  if (config->has_value("output_mesh_file"))
  {
    std::string output_mesh_file =
      config->get_value<std::string>("output_mesh_file");
    std::string extension = ST::GetFilenameLastExtension( output_mesh_file );
    if (!( extension == ".vtp" || extension == ".obj" || extension == ".ply" ))
    {
      KWIVER_CONFIG_FAIL("The output_mesh_file must have a .vtp, .ply, or .obj extension");
    }
  }

  if ( (!config->has_value("output_mesh_file") ||
        config->get_value<std::string>("output_mesh_file") == "") &&
       (!config->has_value("output_volume_file") ||
        config->get_value<std::string>("output_volume_file") == ""))
  {
    LOG_WARN(main_logger, "No output files specified. Nothing will be saved.");
  }

  config_valid =
    validate_optional_output_file("output_mesh_file", *config,
                                 main_logger, false) && config_valid;

  if (!integrate_depth_maps::check_nested_algo_configuration("integrate_depth_maps",
                                                             config))
  {
    KWIVER_CONFIG_FAIL("integration configuration check failed");
  }

#undef KWIVER_CONFIG_FAIL

  return config_valid;
}
} // end namespace

class fuse_depth::priv
{
public:
  priv() {}

  integrate_depth_maps_sptr integrate_algo;
  kv::config_block_sptr config;
  camera_map_sptr camera_map;
  landmark_map_sptr landmark_map;
  vtkSmartPointer<vtkImageData> fused_volume;
  vtkSmartPointer<vtkPolyData> isosurface_mesh;
  double isosurface_threshold = 0.0;

  kv::path_t  input_cameras_directory = "results/krtd";
  kv::path_t  input_depths_directory = "results/depths";
  kv::path_t  input_landmarks_file = "results/landmarks.ply";
  kv::path_t  output_volume_file = "results/volume.vti";
  kv::path_t  output_mesh_file = "results/mesh.vtp";

  enum commandline_mode {SUCCESS, HELP, WRITE, FAIL};

  commandline_mode process_command_line(cxxopts::ParseResult& cmd_args)
  {
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

    if ( cmd_args.count("input-cameras-dir") > 0 )
    {
      input_cameras_directory = cmd_args["input-cameras-dir"].as<std::string>();
      config->set_value("input_cameras_directory", input_cameras_directory);
    }
    if ( cmd_args.count("input-depths-dir") > 0 )
    {
      input_depths_directory = cmd_args["input-depths-dir"].as<std::string>();
      config->set_value("input_depths_directory", input_depths_directory);
    }
    if ( cmd_args.count("input-landmarks-file") > 0 )
    {
      input_landmarks_file = cmd_args["input-landmarks-file"].as<std::string>();
      config->set_value("input_landmarks_file", input_landmarks_file);
    }
    if ( cmd_args.count("output-volume-file") > 0 )
    {
      output_volume_file = cmd_args["output-volume-file"].as<std::string>();
      config->set_value("output_volume_file", output_volume_file);
    }
    if ( cmd_args.count("output-mesh-file") > 0 )
    {
      output_mesh_file = cmd_args["output-mesh-file"].as<std::string>();
      config->set_value("output_mesh_file", output_mesh_file);
    }
    if ( cmd_args.count("isosurface-threshold") > 0 )
    {
      isosurface_threshold = cmd_args["isosurface-threshold"].as<double>();
      config->set_value("isosurface_threshold", isosurface_threshold);
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
    typedef kwiver::tools::kwiver_applet kvt;
    auto config = kvt::find_configuration("applets/fuse_depth.conf");

    config->set_value("input_cameras_directory", input_cameras_directory,
      "Directory to read cameras from.");

    config->set_value("input_landmarks_file", input_landmarks_file,
      "Path to a file to read the landmarks from.");

    config->set_value("input_depths_directory", input_depths_directory,
      "Directory to read the depth maps from.");

    config->set_value("output_volume_file", output_volume_file,
      "Path to a file to write the fused volume. Will be overwritten if "
      "present." );

    config->set_value("output_mesh_file", output_mesh_file,
      "Path to a file to write the extracted isosurface mesh. Will be "
      "overwritten if present." );

    return config;
  }

  void initialize()
  {
    integrate_depth_maps::set_nested_algo_configuration( "integrate_depth_maps",
                                                         config,
                                                         integrate_algo );
  }

  void load_landmarks()
  {
    landmark_map = read_ply_file(input_landmarks_file);
  }

  std::vector<std::string>
  common_camera_and_depth_basenames()
  {
    std::set<std::string> camera_basenames;
    std::set<std::string> depth_basenames;

    kwiversys::Directory cam_dir;
    kwiversys::Directory depth_dir;

    cam_dir.Load(input_cameras_directory);
    unsigned long num_cam_files = cam_dir.GetNumberOfFiles();
    for (unsigned long i = 0; i < num_cam_files; ++i )
    {
      std::string file = cam_dir.GetPath();
      file += "/" + std::string( cam_dir.GetFile( i ) );
      if ( ST::GetFilenameLastExtension( file ) == CAMERA_EXTENSION )
      {
        std::string filename = ST::GetFilenameName(file);
        camera_basenames.insert( ST::GetFilenameWithoutLastExtension(filename) );
      }
    }

    depth_dir.Load(input_depths_directory);
    unsigned long num_depth_files = depth_dir.GetNumberOfFiles();
    for (unsigned long i = 0; i < num_depth_files; ++i )
    {
      std::string file = depth_dir.GetPath();
      file += "/" + std::string( depth_dir.GetFile( i ) );
      if ( ST::GetFilenameLastExtension( file ) == DEPTH_EXTENSION )
      {
        std::string filename = ST::GetFilenameName(file);
        depth_basenames.insert( ST::GetFilenameWithoutLastExtension(filename) );
      }
    }

    std::vector<std::string> union_basenames(std::min(camera_basenames.size(),
                                                      depth_basenames.size()));

    auto last_position = std::set_intersection(camera_basenames.begin(),
                                               camera_basenames.end(),
                                               depth_basenames.begin(),
                                               depth_basenames.end(),
                                               union_basenames.begin());

    union_basenames.resize(last_position-union_basenames.begin());

    return union_basenames;
  }

  bool write_depth_volume()
  {
    if( fused_volume == nullptr)
    {
      LOG_WARN(main_logger, "Fused volume was not set for writing");
      return false;
    }

    if( output_volume_file == "" )
    {
      LOG_INFO(main_logger, "Fused volume output file not set");
      return true;
    }

    auto volume_writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();

    volume_writer->SetFileName(output_volume_file.c_str());
    volume_writer->SetInputData(fused_volume);
    volume_writer->Write();
    return true;
  }

  bool write_isosurface()
  {
    if (output_mesh_file != "")
    {
      std::string extension = ST::GetFilenameLastExtension( output_mesh_file );
      if ( extension == ".ply")
      {
        vtkNew<vtkPLYWriter> mesh_writer;
        mesh_writer->SetFileName(output_mesh_file.c_str());
        mesh_writer->AddInputDataObject(isosurface_mesh);
        mesh_writer->Write();
      }
      else if ( extension == ".obj" )
      {
        vtkNew<vtkOBJWriter> mesh_writer;
        mesh_writer->SetFileName(output_mesh_file.c_str());
        mesh_writer->AddInputDataObject(isosurface_mesh);
        mesh_writer->Write();
      }
      else if ( extension == ".vtp" )
      {
        vtkNew<vtkXMLPolyDataWriter> mesh_writer;
        mesh_writer->SetFileName(output_mesh_file.c_str());
        mesh_writer->AddInputDataObject(isosurface_mesh);
        mesh_writer->Write();
      }
      else
      {
          LOG_WARN(main_logger, "output_mesh_file " << output_mesh_file
                                << " was not a .ply, .obj, or .vtp file");
          return false;
      }
      return true;
    }
    return false;
  }

  void compute_ROI(kv::vector_3d& minpt, kv::vector_3d& maxpt)
  {
    // Convert landmarks to vector
    std::vector<kv::landmark_sptr> landmarks_out;
    landmarks_out.reserve(landmark_map->size());
    for (auto const& l : landmark_map->landmarks())
    {
      landmarks_out.push_back(l.second);
    }

    double bounds[6]; // xmin, xmax, ymin, ymax, zmin, zmax

    compute_robust_ROI(landmarks_out, bounds);

    minpt = {bounds[0], bounds[2], bounds[4]};
    maxpt = {bounds[1], bounds[3], bounds[5]};
  }

  void run_algorithm()
  {
    std::vector<kv::camera_perspective_sptr> cameras_out;
    std::vector<kv::image_container_sptr> depths_out;
    std::vector<kv::image_container_sptr> weights_out;

    std::vector<std::string> shared_basenames
      = common_camera_and_depth_basenames();
    for (auto const& shared : shared_basenames )
    {
      std::string camera_filename = input_cameras_directory + "/" +
                                    shared + CAMERA_EXTENSION;
      std::string depth_filename = input_depths_directory + "/" +
                                    shared + DEPTH_EXTENSION;
      auto cam = read_krtd_file(camera_filename);

      kv::bounding_box<int> crop;
      kv::image_container_sptr depth, weight, uncertainty, color;
      load_depth_map(depth_filename, crop, depth, weight, uncertainty, color);
      depths_out.push_back(depth);
      weights_out.push_back(weight);
      cameras_out.push_back(crop_camera(cam, crop));
    }

    kv::vector_3d minpt, maxpt;
    // Compute the extents from the loaded landmarks
    compute_ROI(minpt, maxpt);

    kv::image_container_sptr volume;
    kv::vector_3d spacing;
    integrate_algo->integrate(minpt, maxpt,
                              depths_out, weights_out, cameras_out,
                              volume, spacing);

    fused_volume = volume_to_vtk(volume, minpt, spacing);
    if (output_mesh_file != "")
    {
      vtkNew<vtkFlyingEdges3D> contour_filter;
      contour_filter->SetInputData(fused_volume);
      contour_filter->SetNumberOfContours(1);
      contour_filter->SetValue(0, isosurface_threshold);
      //Declare which table will be use for the contour
      contour_filter->SetInputArrayToProcess(0, 0, 0,
        vtkDataObject::FIELD_ASSOCIATION_POINTS,
        "reconstruction_scalar");
      contour_filter->Update();

      isosurface_mesh = contour_filter->GetOutput();
    }
  }
};

// ----------------------------------------------------------------------------
int
fuse_depth::
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

    if(d->integrate_algo == nullptr)
    {
      d->initialize();
    }

    if(d->landmark_map == nullptr)
    {
      d->load_landmarks();
      if(d->landmark_map == nullptr)
      {
        LOG_ERROR(main_logger, "Error loading landmarks.");
        return EXIT_FAILURE;
      }
    }

    d->run_algorithm();

    if(!d->write_depth_volume())
    {
      LOG_ERROR(main_logger, "Error writing volume.");
      return EXIT_FAILURE;
    }
    d->write_isosurface();

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
fuse_depth::
add_command_options()
{
  m_cmd_options->custom_help( wrap_text( "[options]\n" ) );

  m_cmd_options->positional_help(
    "\n  input-cameras-dir - name of the directory containing the krtd camera files"
    "(default: " + d->input_cameras_directory + ")"
    "\n  input-depths-dir - name of the directory to read depth maps from "
    "(default: " + d->input_depths_directory + ")");

  m_cmd_options->add_options()
    ( "h,help",     "Display applet usage" )
    ( "c,config",   "Configuration file for tool", cxxopts::value<std::string>() )
    ( "o,output-config",
      "Output a configuration. This may be seeded with a "
      "configuration file from -c/--config.",
      cxxopts::value<std::string>() )
    ( "l,input-landmarks-file", "3D sparse features (default: " +
      d->input_landmarks_file + ")", cxxopts::value<std::string>() )
    ( "m,output-mesh-file", "Write out isocontour mesh to file (default: " +
      d->output_mesh_file + ")",
      cxxopts::value<std::string>())
    ( "v,output-volume-file",
      "Write out integrated integrated depth data to file (default: " +
      d->output_volume_file +")",
      cxxopts::value<std::string>())
    ( "t,isosurface-threshold", "isosurface extraction threshold (default: " +
      std::to_string(d->isosurface_threshold) + ")." , cxxopts::value<double>() )

    // positional parameters
    ( "input-cameras-dir", "Camera location data", cxxopts::value<std::string>() )
    ( "input-depths-dir", "Output directory for depth maps",
      cxxopts::value<std::string>())
    ;

    m_cmd_options->parse_positional({ "input-cameras-dir",
                                      "input-depths-dir"});
}

// ============================================================================
fuse_depth::
fuse_depth()
 : d(new priv())
{ }

fuse_depth::
~fuse_depth() = default;


} } } // end namespace
