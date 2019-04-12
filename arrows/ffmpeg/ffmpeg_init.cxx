/*ckwg +29
* Copyright 2018-2019 by Kitware, Inc.
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

//-----------------------------------------------------------------------------
// some versions of FFMPEG require this definition before including
// the headers for C++ compatibility
#define __STDC_CONSTANT_MACROS

#include "ffmpeg_init.h"


#include <mutex>

extern "C" {
#include <libavformat/avformat.h>
}

#include <vital/logger/logger.h>
#include <vital/util/string.h>

std::mutex ffmpeg_init_mutex;

static auto ffmpeg_logger = kwiver::vital::get_logger( "arrows.ffmpeg" );

//-----------------------------------------------------------------------------
void
ffmpeg_kwiver_log_callback(void* ptr, int level, const char* fmt, va_list vl)
{
  static int print_prefix = 0;
  char line[1024];
  av_log_format_line(ptr, level, fmt, vl, line, sizeof(line), &print_prefix);
  std::string msg(line);
  kwiver::vital::right_trim(msg);
  switch(level)
  {
    case AV_LOG_PANIC:
    case AV_LOG_FATAL:
      LOG_ERROR(ffmpeg_logger, msg);
      break;
    case AV_LOG_ERROR:
    case AV_LOG_WARNING:
      LOG_WARN(ffmpeg_logger, msg);
      break;
    case AV_LOG_INFO:
      LOG_INFO(ffmpeg_logger, msg);
      break;
    case AV_LOG_VERBOSE:
    case AV_LOG_DEBUG:
      LOG_DEBUG(ffmpeg_logger, msg);
      break;
    default:
      break;
  };

}

//-----------------------------------------------------------------------------
void ffmpeg_init()
{
  std::lock_guard< std::mutex > lock(ffmpeg_init_mutex);
  static bool initialized = false;
  if ( ! initialized ) {
    av_register_all();
    av_log_set_callback(ffmpeg_kwiver_log_callback);
    initialized = true;
  }
}
