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
