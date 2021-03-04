// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file collate_process.h
 *
 * \brief Declaration of the collate process.
 */

#ifndef SPROKIT_PROCESSES_FLOW_COLLATE_PROCESS_H
#define SPROKIT_PROCESSES_FLOW_COLLATE_PROCESS_H

#include "processes_flow_export.h"

#include <sprokit/pipeline/process.h>

namespace sprokit {

class PROCESSES_FLOW_NO_EXPORT collate_process
  : public process
{
public:
  PLUGIN_INFO( "collate",
               "Collates data from multiple worker processes," )

  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  collate_process( kwiver::vital::config_block_sptr const& config );
  /**
   * \brief Destructor.
   */
  ~collate_process();

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
  properties_t _properties() const override;

  /**
   * \brief Input port information.
   *
   * \param port The port to get information about.
   *
   * \returns Information about an input port.
   */
  void input_port_undefined( port_t const& port ) override;

private:
  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // SPROKIT_PROCESSES_FLOW_COLLATE_PROCESS_H
