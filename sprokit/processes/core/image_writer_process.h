// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_IMAGE_WRITER_PROCESS_H_
#define _KWIVER_IMAGE_WRITER_PROCESS_H_

#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver
{

// ----------------------------------------------------------------
/**
 * \class image_writer_process
 *
 * \brief Reads a series of images
 *
 * \iports
 * \iport{image}
 * \iport{timetamp}
 *
 */
class KWIVER_PROCESSES_NO_EXPORT image_writer_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "image_writer",
               "Write image to disk." )

  image_writer_process( kwiver::vital::config_block_sptr const& config );
  virtual ~image_writer_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class image_writer_process

}  // end namespace

#endif // _KWIVER_IMAGE_WRITER_PROCESS_H_
