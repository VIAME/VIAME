// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_PROCESSES_REFINE_DETECTIONS_PROCESS_H
#define ARROWS_PROCESSES_REFINE_DETECTIONS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_export.h"

#include <vital/config/config_block.h>

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
class KWIVER_PROCESSES_NO_EXPORT refine_detections_process
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

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
};

} // end namespace

#endif /* ARROWS_PROCESSES_REFINE_DETECTIONS_PROCESS_H */
