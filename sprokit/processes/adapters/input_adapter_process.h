// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to input adapter process.
 */

#ifndef PROCESS_INPUT_ADAPTER_PROCESS_H
#define PROCESS_INPUT_ADAPTER_PROCESS_H

#include <sprokit/processes/adapters/kwiver_adapter_export.h>

#include <sprokit/pipeline/process.h>

#include "adapter_base.h"

namespace kwiver {

// ----------------------------------------------------------------
class KWIVER_ADAPTER_EXPORT input_adapter_process
  : public sprokit::process,
    public adapter::adapter_base
{
public:
  PLUGIN_INFO( "input_adapter",
               "Source process for embedded pipeline.\n\n"
               "Pushes data items into pipeline ports. "
               "Ports are dynamically created as needed based on connections specified in the pipeline file." )

  // -- CONSTRUCTORS --
  input_adapter_process( kwiver::vital::config_block_sptr const& config );
  virtual ~input_adapter_process();

  // Process interface
  void _step() override;

  /**
   * @brief Return list of active ports.
   *
   * This method returns the list of currently active ports and
   * associated port info items.
   *
   * @return List of port names and info.
   */
  adapter::ports_info_t get_ports();

private:

  // This is used to intercept connections and make ports JIT
  void output_port_undefined( sprokit::process::port_t const& port) override;

}; // end class input_adapter_process

} // end namespace

#endif /* PROCESS_INPUT_ADAPTER_PROCESS_H */
