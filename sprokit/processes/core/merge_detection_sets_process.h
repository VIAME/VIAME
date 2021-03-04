// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef _KWIVER_MERGE_DETECTION_SETS_PROCESS_H_
#define _KWIVER_MERGE_DETECTION_SETS_PROCESS_H_

#include "kwiver_processes_export.h"

#include <sprokit/pipeline/process.h>

namespace kwiver
{

class KWIVER_PROCESSES_NO_EXPORT merge_detection_sets_process
  : public sprokit::process
{

public:
  PLUGIN_INFO( "merge_detection_sets",
               "Merge multiple input detection sets into one output set.\n\n"
               "This process will accept one or more input ports of detected_object_set "
               "type. They will all be added to the output detection set. "
               "The input port names do not matter since they will be connected "
               "upon connection.")

  merge_detection_sets_process( vital::config_block_sptr const& config );
  virtual ~merge_detection_sets_process();

protected:
  virtual void _configure();
  virtual void _step();
  virtual void _init();

  void input_port_undefined( port_t const& port ) override;

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; //end class merge_detection_sets_process

} // end namespace

#endif /*_KWIVER_MERGE_DETECTION_SETS_PROCESS_H_*/
