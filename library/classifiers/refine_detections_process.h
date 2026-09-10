// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_PROCESSES_REFINE_DETECTIONS_PROCESS_H
#define ARROWS_PROCESSES_REFINE_DETECTIONS_PROCESS_H

#include <viame/pipeline_framework/process.h>

#include "viame_processes_classifiers_export.h"

#include <viame/algorithm_framework/config/config_block.h>

namespace kwiver {

// ----------------------------------------------------------------
/**
 * @brief Object detection refiner process.
 *
 * \iports
 * \iport{image}
 * \iport{detected_object_set}
 *
 * \oports
 * \oport{detected_object_set}
 */
class VIAME_PROCESSES_CLASSIFIERS_EXPORT refine_detections_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "refine_detections",
               "Refines detections for a given frame," )

  refine_detections_process( kwiver::vital::config_block_sptr const& config );
  virtual ~refine_detections_process();

protected:
  virtual void _configure();
  virtual void _step();
  void _finalize();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
};

} // end namespace

#endif /* ARROWS_PROCESSES_REFINE_DETECTIONS_PROCESS_H */
