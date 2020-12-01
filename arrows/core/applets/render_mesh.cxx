// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
///
/// This tool renders a mesh into an image depth or height map

#include "render_mesh.h"

#include <iostream>
#include <fstream>

#include <vital/algo/image_io.h>
#include <vital/config/config_block.h>
#include <vital/config/config_block_io.h>
#include <vital/config/config_block_formatter.h>
#include <vital/exceptions.h>
#include <vital/io/mesh_io.h>
#include <vital/io/camera_io.h>
#include <vital/util/get_paths.h>
#include <vital/util/transform_image.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <arrows/core/mesh_operations.h>
#include <arrows/core/render_mesh_depth_map.h>

namespace kwiver {
namespace arrows {
namespace core {

namespace {

// Global options
std::string opt_out_config;     // output config file name
int opt_width;
int opt_height;

// ------------------------------------------------------------------
static kwiver::vital::config_block_sptr default_config()
{
  kwiver::vital::config_block_sptr config =
    kwiver::vital::config_block::empty_config( "render-mesh-tool" );

  config->set_value( "image_io:type", "vxl",
                     "Implementation for image writer" );

  kwiver::vital::algo::image_io::get_nested_algo_configuration(
    "image_io", config, kwiver::vital::algo::image_io_sptr() );

  return config;
}

} // end namespace

// ----------------------------------------------------------------------------
template <typename T>
kwiver::vital::image_of<uint8_t>
stretch_to_byte_image(kwiver::vital::image_of<T> const& in_image)
{
  // get the range of finite values in the image
  double min_v = std::numeric_limits<T>::infinity();
  double max_v = -std::numeric_limits<T>::infinity();
  kwiver::vital::foreach_pixel(in_image, [&min_v, &max_v](T p)
  {
    if( std::isfinite(p) )
    {
      min_v = std::min(min_v, p);
      max_v = std::max(max_v, p);
    }
  });

  // map the range of finite values to [1, 255] and use 0 for all non-finite values
  T scale = 254.0 / (max_v - min_v);
  T offset = 1.0 - min_v * scale;
  kwiver::vital::image_of<uint8_t> byte_image(in_image.width(), in_image.height());
  kwiver::vital::transform_image(in_image, byte_image,
                                 [scale, offset](T p) -> uint8_t
  {
    if (std::isfinite(p))
    {
      return static_cast<uint8_t>(p * scale + offset);
    }
    return 0;
  });

  return byte_image;
}

// ----------------------------------------------------------------------------
void
render_mesh::
add_command_options()
{
  m_cmd_options->custom_help( wrap_text(
       "This tool renders a mesh into a depth or height image\n"
       "\n"
       "Usage: kwiver " + applet_name() + " [options] mesh camera image"
          ) );

  m_cmd_options->positional_help( "\n   mesh - Mesh file name.\n"
                                  "   camera - camera file name.\n"
                                  "   output - Output image file name");

  m_cmd_options->add_options()
    ( "h,help",        "Display usage information" )
    ( "c",             "Configuration file for tool" )
    ( "output-config", "Dump configuration for tool", cxxopts::value<std::string>() )
    ( "x",             "Output image width", cxxopts::value<int>()->default_value( "1920" ) )
    ( "y",             "Output image height", cxxopts::value<int>()->default_value( "1080" ) )
    ( "height-map",    "Render a height map instead of a depth map" )
    ( "byte",          "Render as a byte image with scaled range" )

    // positional parameters
    ( "mesh-file",    "Mesh file name", cxxopts::value<std::string>() )
    ( "camera-file",  "Camera file name", cxxopts::value<std::string>() )
    ( "output-image", "Output image file name", cxxopts::value<std::string>() )
    ;

    m_cmd_options->parse_positional({"mesh-file", "camera-file", "output-image"});
}

// ----------------------------------------------------------------
/** Main entry. */
int
render_mesh::
run()
{
  auto& cmd_args = command_args();

  if ( cmd_args["help"].as<bool>() )
  {
    std::cout << m_cmd_options->help();
    return EXIT_SUCCESS;
  }

  // If we are not writing out the config, then all positional file
  // names are required.
  if ( cmd_args.count("output-config") == 0 )
  {
    if ( ( cmd_args.count("mesh-file") == 0 ) ||
         ( cmd_args.count("camera-file") == 0 ) ||
         ( cmd_args.count("output-image") == 0 ) )
    {
      std::cout << "Missing file name.\n"
                << "Usage: " << applet_name()
                << " mesh-file camera-file output-image\n"
                << std::endl;

      return EXIT_FAILURE;
    }
  }

  kwiver::vital::algo::image_io_sptr image_writer;
  kwiver::vital::config_block_sptr config = default_config();
  // If --config given, read in config file, merge in with default just generated
  if ( cmd_args.count("c") > 0 )
  {
    config->merge_config( kwiver::vital::read_config_file( cmd_args["c"].as<std::string>() ) );
  }

  kwiver::vital::algo::image_io::set_nested_algo_configuration( "image_io", config, image_writer );
  kwiver::vital::algo::image_io::get_nested_algo_configuration( "image_io", config, image_writer );

  // Check to see if we are to dump config
  if ( cmd_args.count("output-config") > 0 )
  {
    opt_out_config = cmd_args["output-config"].as<std::string>();
    std::ofstream fout( opt_out_config );
    if( ! fout )
    {
      std::cout << "Couldn't open \"" << opt_out_config << "\" for writing.\n";
      return EXIT_FAILURE;
    }

    kwiver::vital::config_block_formatter fmt( config );
    fmt.print( fout );
    std::cout << "Wrote config to \"" << opt_out_config << "\". Exiting.\n";
    return EXIT_SUCCESS;
  }

  if( !kwiver::vital::algo::image_io::check_nested_algo_configuration( "image_io", config ) )
  {
    std::cerr << "Invalid image_io config" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string mesh_file = cmd_args["mesh-file"].as<std::string>();
  const std::string camera_file = cmd_args["camera-file"].as<std::string>();
  const std::string image_file = cmd_args["output-image"].as<std::string>();

  opt_width = cmd_args["x"].as<int>();
  opt_height = cmd_args["y"].as<int>();

  std::cout << "Reading Mesh" << std::endl;
  auto mesh = kwiver::vital::read_mesh(mesh_file);
  std::cout << "Reading Camera" << std::endl;
  auto camera = kwiver::vital::read_krtd_file(camera_file);
  auto K = std::make_shared<kwiver::vital::simple_camera_intrinsics>(*camera->intrinsics());
  K->set_image_width(opt_width);
  K->set_image_height(opt_height);
  std::dynamic_pointer_cast<kwiver::vital::simple_camera_perspective>(camera)->set_intrinsics(K);

  if ( mesh->faces().regularity() != 3 )
  {
    std::cout << "Triangulating Mesh" << std::endl;
    kwiver::arrows::core::mesh_triangulate(*mesh);
  }
  std::cout << "Clipping Mesh to Camera Frustum" << std::endl;
  kwiver::arrows::core::clip_mesh(*mesh, *camera);

  std::cout << "Rendering" << std::endl;
  vital::image_container_sptr image;
  if ( cmd_args["height-map"].as<bool>() )
  {
    image = render_mesh_height_map(mesh, camera);
  }
  else
  {
    image = render_mesh_depth_map(mesh, camera);
  }

  if ( cmd_args["byte"].as<bool>() )
  {
    std::cout << "Converting to byte image" << std::endl;
    kwiver::vital::image_of<double> d_image(image->get_image());
    image = std::make_shared<kwiver::vital::simple_image_container>(
              stretch_to_byte_image(d_image));
  }

  std::cout << "Saving" << std::endl;
  image_writer->save(image_file, image);
  return EXIT_SUCCESS;
}

} } } // end namespace
