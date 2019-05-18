/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

#include "vital/types/color_space.h"

#include <algorithm>
#include <map>

/// \brief Converts a string to a known color space if possible.
kwiver::vital::color_space
kwiver::vital::string_to_color_space( const std::string& str )
{
  std::string lowercase_str = str;

  std::transform( lowercase_str.begin(),
                  lowercase_str.end(),
                  lowercase_str.begin(),
                  ::tolower );

  // Manually create a short branch tree. Another common way to create
  // a string to enum correspondance would be to use a static std::map,
  // but that would incur a tiny amount of extra run-time memory.
  if( lowercase_str.size() < 3 )
  {
    return kwiver::vital::INVALID_CS;
  }

  if( lowercase_str[0] < 'k' )
  {
    if( lowercase_str == "bgr" )
    {
      return kwiver::vital::BGR;
    }
    if( lowercase_str == "cmyk" )
    {
      return kwiver::vital::CMYK;
    }
    if( lowercase_str == "hls" )
    {
      return kwiver::vital::HLS;
    }
    if( lowercase_str == "hsl" )
    {
      return kwiver::vital::HSL;
    }
    if( lowercase_str == "hsv" )
    {
      return kwiver::vital::HSV;
    }
  }
  else
  {
    if( lowercase_str == "lab" )
    {
      return kwiver::vital::Lab;
    }
    if( lowercase_str == "luv" )
    {
      return kwiver::vital::Luv;
    }
    if( lowercase_str == "rgb" )
    {
      return kwiver::vital::RGB;
    }
    if( lowercase_str == "xyz" )
    {
      return kwiver::vital::XYZ;
    }
    if( lowercase_str == "ycrcb" )
    {
      return kwiver::vital::YCrCb;
    }
    if( lowercase_str == "ycbcr" )
    {
      return kwiver::vital::YCbCr;
    }
  }

  return kwiver::vital::INVALID_CS;
}

/// \brief Converts a known color space to a string
std::string
kwiver::vital::color_space_to_string( const kwiver::vital::color_space cs )
{

  const static std::map< kwiver::vital::color_space, std::string > mapping =
   { { kwiver::vital::INVALID_CS, "INVALID" },
     { kwiver::vital::BGR, "BGR" },
     { kwiver::vital::CMYK, "CMYK" },
     { kwiver::vital::HLS, "HLS" },
     { kwiver::vital::HSL, "HSL" },
     { kwiver::vital::HSV, "HSV" },
     { kwiver::vital::Lab, "Lab" },
     { kwiver::vital::Luv, "Luv" },
     { kwiver::vital::RGB, "RGB" },
     { kwiver::vital::XYZ, "XYZ" },
     { kwiver::vital::YCrCb, "YCrCb" },
     { kwiver::vital::YCbCr, "YCbCr" } };

  return mapping.at( cs );
}
