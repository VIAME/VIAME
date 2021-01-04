// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file distribute_process.h
 *
 * \brief Declaration of the distribute process.
 */

#ifndef SPROKIT_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H
#define SPROKIT_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H

#include "processes_flow_export.h"

#include <sprokit/pipeline/process.h>

namespace sprokit {

class PROCESSES_FLOW_NO_EXPORT distribute_process
  : public process
{
public:
  PLUGIN_INFO( "distribute",
               "Distributes data to multiple worker processes." )

  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  distribute_process( kwiver::vital::config_block_sptr const& config );
  /**
   * \brief Destructor.
   */
  ~distribute_process();

protected:
  /**
   * \brief Initialize the process.
   */
  void _init() override;

  /**
   * \brief Reset the process.
   */
  void _reset() override;

  /**
   * \brief Step the process.
   */
  void _step() override;

  /**
   * \brief The properties on the process.
   */
  properties_t _properties() const;

  /**
   * \brief Output port information.
   *
   * \param port The port to get information about.
   *
   * \returns Information about an output port.
   */
  void output_port_undefined( port_t const& port ) override;

private:
  class priv;
  std::unique_ptr< priv > d;
};

}

#endif // SPROKIT_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H
