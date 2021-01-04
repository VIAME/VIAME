// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_DATA_DEPENDENT_PROCESS_H
#define SPROKIT_PROCESSES_EXAMPLES_DATA_DEPENDENT_PROCESS_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process.h>

/**
 * \file data_dependent_process.h
 *
 * \brief Declaration of the data dependent process.
 */

namespace sprokit {

/**
 * \class data_dependent_process
 *
 * \brief A process with a data dependent port.
 *
 * \process A process with a data dependent port.
 *
 * \configs
 *
 * \config{reject} Whether to reject the set type or not.
 * \config{set_on_configure} Whether to set the type on configure or not.
 *
 * \oports
 *
 * \oport{output} A data dependent output port.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT data_dependent_process
  : public process
{
public:
  PLUGIN_INFO( "data_dependent",
               "A process with a data dependent type" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  data_dependent_process(kwiver::vital::config_block_sptr const& config);
  /**
   * \brief Destructor.
   */
  ~data_dependent_process();
protected:
  /**
   * \brief Configure the process.
   */
  void _configure() override;
  /**
   * \brief Step the process.
   */
  void _step() override;
  /**
   * \brief Reset the process.
   */
  void _reset() override;
  /**
   * \brief Set the type for an output port.
   *
   * \param port The name of the port.
   * \param new_type The type of the connected port.
   *
   * \returns True if the type can work, false otherwise.
   */
  bool _set_output_port_type(port_t const& port, port_type_t const& new_type) override;

private:
  void make_ports();

  class priv;
  std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_DATA_DEPENDENT_PROCESS_H
