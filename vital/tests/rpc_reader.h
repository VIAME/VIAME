// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_TESTS_RPC_READER_H_
#define VITAL_TESTS_RPC_READER_H_

#include <fstream>
#include <iostream>

#include <vital/types/camera_rpc.h>

// Reads RPC coeffs from file and creates a camera from them
kwiver::vital::simple_camera_rpc read_rpc( std::string filename )
{
  kwiver::vital::rpc_matrix rpc_coeffs = kwiver::vital::rpc_matrix::Zero();
  kwiver::vital::vector_3d world_scale;
  kwiver::vital::vector_3d world_offset;
  kwiver::vital::vector_2d image_scale;
  kwiver::vital::vector_2d image_offset;
  kwiver::vital::vector_2i image_dimension(0, 0);

  std::ifstream rpc_file;
  rpc_file.open( filename );

  unsigned int line_idx = 0;
  while (! rpc_file.eof() )
  {
    std::string line;
    std::getline( rpc_file, line );

    std::stringstream ss(line);
    double value;
    unsigned int word_idx = 0;
    while ( ss >> value )
    {
      if ( line_idx < 4 )
      {
        rpc_coeffs( line_idx, word_idx ) = value;
      }
      else
      {
        switch ( line_idx )
        {
          case 4:
            world_scale( word_idx ) = value;
            break;
          case 5:
            world_offset( word_idx ) = value;
            break;
          case 6:
            image_scale( word_idx ) = value;
            break;
          case 7:
            image_offset( word_idx ) = value;
            break;
          case 8:
            image_dimension( word_idx ) = static_cast<int>(value);
            break;
        }
      }
      word_idx++;
    }
    line_idx++;
  }

  rpc_file.close();

  return kwiver::vital::simple_camera_rpc( world_scale, world_offset,
                                           image_scale, image_offset,
                                           rpc_coeffs, image_dimension[0], image_dimension[1]);
}

#endif // VITAL_TESTS_RPC_READER_H_
