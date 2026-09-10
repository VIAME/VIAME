// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_PROCESSES_IMAGE_OBJECT_DETECTOR_PROCESS_H
#define ARROWS_PROCESSES_IMAGE_OBJECT_DETECTOR_PROCESS_H

#include <viame/pipeline_framework/process.h>

#include "viame_processes_object_detectors_export.h"

#include <viame/algorithm_framework/config/config_block.h>

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
class VIAME_PROCESSES_OBJECT_DETECTORS_EXPORT image_object_detector_process
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
