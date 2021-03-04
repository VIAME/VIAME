// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_FLOW_DEPENDENT_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_FLOW_DEPENDENT_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file flow_dependent_process.h
 *
 * \brief Declaration of the flow dependent process.
 */

namespace sprokit {

/**
 * \class flow_dependent_process
 *
 * \brief A process with flow dependent ports.
 *
 * \process A process with flow dependent ports.
 *
 * \configs
 *
 * \config{reject} Whether to reject the set type or not.
 *
 * \iports
 *
 * \iport{input} A flow dependent input port.
 *
 * \oports
 *
 * \oport{output} A flow dependent output port.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT flow_dependent_process : public process {
public:
  PLUGIN_INFO( "flow_dependent",
               "A process with a flow dependent type" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  flow_dependent_process(kwiver::vital::config_block_sptr const &config);
  /**
   * \brief Destructor.
   */
  ~flow_dependent_process();

protected:
  /**
   * \brief Reset the process.
   */

  void _reset() override;
  /**
   * \brief Step the process.
   */
  void _step() override;

  /**
   * \brief Set the type for an input port.
   *
   * \param port The name of the port.
   * \param new_type The type of the connected port.
   *
   * \returns True if the type can work, false otherwise.
   */
  bool _set_input_port_type(port_t const &port, port_type_t const &new_type) override;

  /**
   * \brief Set the type for an output port.
   *
   * \param port The name of the port.
   * \param new_type The type of the connected port.
   *
   * \returns True if the type can work, false otherwise.
   */
  bool _set_output_port_type(port_t const &port, port_type_t const &new_type) override;

private:
  void make_ports();

  class priv;
  std::unique_ptr<priv> d;
};
}

#endif // SPROKIT_PROCESSES_EXAMPLES_FLOW_DEPENDENT_PROCESS_H
