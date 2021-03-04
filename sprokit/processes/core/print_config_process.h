// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_PROCESSES_PRINT_CONFIG_PROCESS_H
#define ARROWS_PROCESSES_PRINT_CONFIG_PROCESS_H

#include <sprokit/pipeline/process.h>

#include "kwiver_processes_export.h"

#include <vital/config/config_block.h>

namespace kwiver {

// ----------------------------------------------------------------
/**
 * @brief Image object detector process.
 *
 */
class KWIVER_PROCESSES_NO_EXPORT print_config_process
  : public sprokit::process
{
public:
  PLUGIN_INFO( "print_config",
               "Print process configuration.\n\n"
               "This process is a debugging aide and performs no other function in a pipeline. "
               "The supplied configuration is printed when it is presented to the process. "
               "All ports connections to the process are accepted and the supplied data is "
               "taken from the port and discarded. This process produces no outputs and "
               "has no output ports.")

  print_config_process( kwiver::vital::config_block_sptr const& config );
  virtual ~print_config_process();

protected:
  virtual void _configure();
  virtual void _step();

  // This is used to intercept connections and make ports JIT
  void input_port_undefined(port_t const& port) override;

private:
  class priv;
  const std::unique_ptr<priv> d;
}; // end class object_detector_process

} // end namespace

#endif // ARROWS_PROCESSES_PRINT_CONFIG_PROCESS_H
