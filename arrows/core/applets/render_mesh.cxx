/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#include <kwiversys/CommandLineArguments.hxx>

namespace kwiver {
namespace arrows {
namespace core {

namespace {

// Global options
bool        opt_help( false );
std::string opt_config;         // config file name
std::string opt_out_config;     // output config file name
int opt_width = 1920;           // image width
int opt_height = 1080;          // image height
bool opt_height_map = false;    // render a height map instead of a depth map
bool opt_byte_img = false;      // render as a byte image with stretched range

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



}

typedef kwiversys::CommandLineArguments argT;


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
usage( std::ostream& outstream ) const
{
  outstream << "This tool renders a mesh into a depth or height image\n"
            << "\n"
            << "Usage: kwiver " << applet_name() << " [options] mesh camera image\n"
            << "\n"
            << "Options are:\b"
            << "  -h | --help            displays usage information\n"
            << "  --config | -c  FILE    Configuration for tool\n"
            << "  --output-config  FILE  Dump configuration to file\n"
    ;
}


// ----------------------------------------------------------------
/** Main entry. */
int
render_mesh::
run( const std::vector<std::string>& argv )
{
  kwiversys::CommandLineArguments arg;

  arg.Initialize( argv );
  arg.StoreUnusedArguments( true );

  arg.AddArgument( "--help",        argT::NO_ARGUMENT, &opt_help, "Display usage information" );
  arg.AddArgument( "-h",              argT::NO_ARGUMENT, &opt_help, "Display usage information" );
  arg.AddArgument( "-c",            argT::SPACE_ARGUMENT, &opt_config, "Configuration file for tool" );
  arg.AddArgument( "--output-config", argT::SPACE_ARGUMENT, &opt_out_config, "Dump configuration for tool" );
  arg.AddArgument( "-x",            argT::SPACE_ARGUMENT, &opt_width, "Output image width");
  arg.AddArgument( "-y",            argT::SPACE_ARGUMENT, &opt_height, "Output image height");
  arg.AddArgument( "--height-map",  argT::NO_ARGUMENT, &opt_height_map, "Render a height map instead of a depth map" );
  arg.AddArgument( "--byte",        argT::NO_ARGUMENT, &opt_byte_img, "Render as a byte image with scaled range" );

  if ( ! arg.Parse() )
  {
    std::cerr << "Problem parsing arguments" << std::endl;
    return EXIT_FAILURE;
  }

  if ( opt_help )
  {
    usage( std::cerr );
    return EXIT_SUCCESS;
  }

  char** newArgv = 0;
  int newArgc = 0;
  arg.GetUnusedArguments(&newArgc, &newArgv);

  if( ( newArgc < 4 ) && ( opt_out_config.empty()) )
  {
    std::cout << "Missing file name.\n"
      << "Usage: " << newArgv[0] << " mesh-file camera-file output-image\n" << std::endl;

      return EXIT_FAILURE;
  }

  kwiver::vital::algo::image_io_sptr image_writer;
  kwiver::vital::config_block_sptr config = default_config();
  // If --config given, read in config file, merge in with default just generated
  if( ! opt_config.empty() )
  {
    config->merge_config( kwiver::vital::read_config_file( opt_config ) );
  }

  kwiver::vital::algo::image_io::set_nested_algo_configuration( "image_io", config, image_writer );
  kwiver::vital::algo::image_io::get_nested_algo_configuration( "image_io", config, image_writer );
  // Check to see if we are to dump config
  if ( ! opt_out_config.empty() )
  {
    std::ofstream fout( opt_out_config.c_str() );
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


  std::string mesh_file = newArgv[1];
  std::string camera_file = newArgv[2];
  std::string image_file = newArgv[3];

  arg.DeleteRemainingArguments(newArgc, &newArgv);
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
  std::cout << "Rendering" << std::endl;
  vital::image_container_sptr image;
  if ( opt_height_map )
  {
    image = render_mesh_height_map(mesh, camera);
  }
  else
  {
    image = render_mesh_depth_map(mesh, camera);
  }

  if ( opt_byte_img )
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
