// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to the filter process.
 */

#ifndef SPROKIT_PROCESSES_CODE_DETECTED_OBJECT_FILTER_PROCESS_H
#define SPROKIT_PROCESSES_CODE_DETECTED_OBJECT_FILTER_PROCESS_H

#include <viame/pipeline_framework/process.h>

#include "viame_processes_classifiers_export.h"

#include <viame/algorithm_framework/config/config_block.h>

namespace kwiver
{

// ----------------------------------------------------------------
class VIAME_PROCESSES_CLASSIFIERS_EXPORT detected_object_filter_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "detected_object_filter",
               "Filters sets of detected objects using the "
               "detected_object_filter algorithm." )

  detected_object_filter_process( kwiver::vital::config_block_sptr const& config );
  virtual ~detected_object_filter_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class detected_object_filter_process

} //end namespace

#endif // SPROKIT_PROCESSES_CODE_DETECTED_OBJECT_FILTER_PROCESS_H
