/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

/**
 * \file
 * \brief This file contains the implementation for vital video metadata
 *  utility functions.
 */

#include "video_metadata_util.h"

#include <kwiversys/SystemTools.hxx>


namespace kwiver {
namespace vital {


/// Extract an image file basename from metadata and (if needed) frame number
std::string
basename_from_metadata(video_metadata_sptr md,
                       frame_id_t frame)
{
  typedef kwiversys::SystemTools  ST;

  std::string basename = "frame";
  if( md && md->has( kwiver::vital::VITAL_META_IMAGE_FILENAME ) )
  {
    std::string img_name = md->find( VITAL_META_IMAGE_FILENAME ).as_string();
    basename = ST::GetFilenameWithoutLastExtension( img_name );
  }
  else
  {
    if ( md && md->has( kwiver::vital::VITAL_META_VIDEO_FILENAME ) )
    {
      std::string vid_name = md->find( VITAL_META_VIDEO_FILENAME ).as_string();
      basename = ST::GetFilenameWithoutLastExtension( vid_name );
    }
    char frame_str[6];
    std::snprintf(frame_str, 6, "%05d", static_cast<int>(frame));
    basename += std::string(frame_str);
  }
  return basename;
}


} } // end namespace
