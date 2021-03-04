// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to output adapter.
 */

#ifndef KWIVER_OUTPUT_ADAPTER_H
#define KWIVER_OUTPUT_ADAPTER_H

#include <sprokit/processes/adapters/kwiver_adapter_export.h>

#include "adapter_types.h"
#include "adapter_data_set.h"

namespace kwiver {

class output_adapter_process;

// -----------------------------------------------------------------
/**
 * @brief Output adapter interface to output adapter process.
 *
 * This class represents a user interface to the output end of a
 * sprokit pipeline. An object of this class attaches to the
 * output_adapter_process in a pipeline and provides an API to
 * communicate with that process which is running in another thread.
 */
class KWIVER_ADAPTER_EXPORT output_adapter
{
public:
  output_adapter();
  virtual ~output_adapter();

  /**
   * @brief Connect to named process.
   *
   * The named process is located in the specified pipeline.
   *
   * @param proc Process name.
   * @param pipe Pipeline to search.
   *
   * @return Pointer to adapter base object
   *
   * @throws sprokit::no_such_port_exception if the process is not found
   */
  void connect( sprokit::process::name_t proc, sprokit::pipeline_t pipe );

  /**
   * @brief Return list of ports connected to adapter process.
   *
   * This method returns the list of output ports that are connected
   * to the adapter process.
   *
   * @return List of port names
   */
  sprokit::process::ports_t port_list() const;

  /**
   * @brief Return list of active ports.
   *
   * This method returns the list of currently active ports and
   * associated port info items.
   *
   * @return List of port names and info.
   */
  virtual adapter::ports_info_t get_ports() const;

  /**
   * @brief Send data set to output adapter process.
   *
   * The specified data set is sent to the output adapter process that
   * is currently connected to this object.
   *
   * @returns Data set
   */
  kwiver::adapter::adapter_data_set_t receive();

  /**
   * @brief Is interface queue empty?
   *
   * This method checks to see if there is a pipeline output data set ready.
   *
   * @return \b true if interface queue is full and thread would wait for receive().
   */
  bool empty() const;

private:
  kwiver::output_adapter_process* m_process;
  kwiver::adapter::interface_ref_t m_interface_queue;
}; // end class output_adapter

} // end namespace

#endif // KWIVER_OUTPUT_ADAPTER_H
