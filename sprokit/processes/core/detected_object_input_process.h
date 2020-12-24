// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for detected_object_set_input process
 */

#ifndef _KWIVER_DETECTED_OBJECT_INPUT_PROCESS_H
#define _KWIVER_DETECTED_OBJECT_INPUT_PROCESS_H
#include <sprokit/pipeline/process.h>
#include "kwiver_processes_export.h"

#include <memory>

namespace kwiver {

// ----------------------------------------------------------------
class KWIVER_PROCESSES_NO_EXPORT detected_object_input_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "detected_object_input",
               "Reads detected object sets from an input file.\n\n"
               "Detections read from the input file are grouped into sets for each "
               "image and individually returned." )

  detected_object_input_process( kwiver::vital::config_block_sptr const& config );
  virtual ~detected_object_input_process();

protected:
  virtual void _configure();
  virtual void _init();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;
}; // end class detected_object_input_process

} // end namespace

#endif // _KWIVER_DETECTED_OBJECT_INPUT_PROCESS_H
