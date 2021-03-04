// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to output adapter process.
 */

#ifndef KWIVER_OUTPUT_ADAPTER_PROCESS_H
#define KWIVER_OUTPUT_ADAPTER_PROCESS_H

#include <sprokit/processes/adapters/kwiver_adapter_export.h>

#include <sprokit/pipeline/process.h>

#include "adapter_base.h"

namespace kwiver {

// ----------------------------------------------------------------
class KWIVER_ADAPTER_EXPORT output_adapter_process
  : public sprokit::process,
    public adapter::adapter_base
{
public:
  PLUGIN_INFO( "output_adapter",
               "Sink process for embedded pipeline.\n\n"
               "Accepts data items from pipeline ports. "
               "Ports are dynamically created as needed based on "
               "connections specified in the pipeline file." )

  // -- CONSTRUCTORS --
  output_adapter_process( kwiver::vital::config_block_sptr const& config );
  virtual ~output_adapter_process();

  // Process interface
  void _step() override;
  void _finalize() override;

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
  void _configure() override;

  // This is used to intercept connections and make ports JIT
  void input_port_undefined(port_t const& port) override;

  class priv;
  const std::unique_ptr<priv> d;

}; // end class output_adapter_process

} // end namespace

#endif // KWIVER_OUTPUT_ADAPTER_PROCESS_H
