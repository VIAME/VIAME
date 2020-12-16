// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

//-----------------------------------------------------------------------------
// some versions of FFmpeg require this definition before including
// the headers for C++ compatibility
#define __STDC_CONSTANT_MACROS

#include "ffmpeg_init.h"

#include <mutex>

extern "C" {
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
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
    avfilter_register_all();
    av_log_set_callback(ffmpeg_kwiver_log_callback);
    initialized = true;
  }
}
