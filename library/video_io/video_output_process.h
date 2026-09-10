// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_VIDEO_OUTPUT_PROCESS_H
#define _KWIVER_VIDEO_OUTPUT_PROCESS_H

#include <viame/pipeline_framework/process.h>
#include "viame_processes_video_io_export.h"

namespace kwiver
{

// -----------------------------------------------------------------------------
/**
 * \class video_output_process
 *
 * \brief Writes a video file and passes the images as a video as output.
 *
 * \iports
 * \iport{image}
 * \iport{timestamp}
 * \iport{metadata}
 */
class VIAME_PROCESSES_VIDEO_IO_NO_EXPORT video_output_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "video_output",
               "Writes video file based on sequential images with optional "
               "metadata per frame." )

  video_output_process( kwiver::vital::config_block_sptr const& config );
  virtual ~video_output_process();

protected:
  virtual void _configure();
  virtual void _init();
  virtual void _step();
  virtual void _finalize();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class video_output_process

}  // end namespace

#endif /* _KWIVER_VIDEO_OUTPUT_PROCESS_H */
