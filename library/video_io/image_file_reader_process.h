// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_IMAGE_FILE_READER_PROCESS_H_
#define _KWIVER_IMAGE_FILE_READER_PROCESS_H_

#include <viame/pipeline_framework/process.h>
#include "viame_processes_video_io_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class image_file_reader_process
 *
 * \brief Reads a series of images
 *
 * \oports
 * \oport{image}
 *
 * \oport{frame}
 * \oport{time}
 *
 * \configs
 *
 * \config{error_mode}  (string)
 * \config{path}  (string)
 * \config{frame_time} (double)
 * \config{image_reader}  (string)
 *
 */
class VIAME_PROCESSES_VIDEO_IO_NO_EXPORT image_file_reader_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "image_file_reader",
               "Reads an image file given the file name." )

  image_file_reader_process( kwiver::vital::config_block_sptr const& config );
  virtual ~image_file_reader_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class image_file_reader_process

}  // end namespace

#endif // _KWIVER_IMAGE_FILE_READER_PROCESS_H_
