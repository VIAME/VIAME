// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_PROCESSES_IMAGE_OBJECT_DETECTOR_PROCESS_H
#define ARROWS_PROCESSES_IMAGE_OBJECT_DETECTOR_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_export.h"

#include <vital/config/config_block.h>

namespace kwiver {

// ----------------------------------------------------------------
/**
 * @brief Image object detector process.
 *
 * \iports
 * \iport{image}
 *
 * \oports
 *
 * \oport{detected_object_set}
 */
class KWIVER_PROCESSES_NO_EXPORT image_object_detector_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "image_object_detector",
               "Apply selected image object detector algorithm to incoming images." )

  image_object_detector_process( kwiver::vital::config_block_sptr const& config );
  virtual ~image_object_detector_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class object_detector_process

} // end namespace

#endif /* ARROWS_PROCESSES_IMAGE_OBJECT_DETECTOR_PROCESS_H */
