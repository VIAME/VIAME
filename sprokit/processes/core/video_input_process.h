// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_VIDEO_INPUT_PROCESS_H
#define _KWIVER_VIDEO_INPUT_PROCESS_H

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class video_input_process
 *
 * \brief Reads a video file and passes the images as output.
 *
 * \oports
 * \oport{image}
 *
 * \oport{frame}
 * \oport{time}
 * \oport{metadata}
 */
class KWIVER_PROCESSES_NO_EXPORT video_input_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "video_input",
               "Reads video files and produces sequential images with metadata per frame." )

  video_input_process( kwiver::vital::config_block_sptr const& config );
  virtual ~video_input_process();

protected:
  virtual void _configure();
  virtual void _init();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class video_input_process

}  // end namespace

#endif /* _KWIVER_VIDEO_INPUT_PROCESS_H */
