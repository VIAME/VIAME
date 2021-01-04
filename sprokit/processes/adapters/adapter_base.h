// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface file for the adapter case class
 */

#ifndef PROCESS_ADAPTER_BASE_H
#define PROCESS_ADAPTER_BASE_H

#include "adapter_types.h"
#include "adapter_data_set.h"

#include <sprokit/pipeline/process.h>

#include <vital/util/bounded_buffer.h>

#include <set>

namespace kwiver {
namespace adapter {

// ----------------------------------------------------------------
/**
 * \class adapter_base
 *
 * \brief Base class for sprokit external adapters
 *
 * This class contains all common code to support the input and output
 * adapter classes. It is not designed to be polymorphic base class.
 */
class adapter_base
{
public:
  adapter_base();
  virtual ~adapter_base();

  /**
   * @brief Get pointer to our interface queue.
   *
   * This interface queue is how the input process is supplied with
   * data to put in the pipeline. It also works to get data from the
   * queue that has been added by the output process.
   *
   * @return Pointer to interface queue.
   */
  interface_ref_t get_interface_queue();

  /**
   * @brief Get list of connected ports.
   *
   * This method returns a copy of the list of connected ports.
   *
   * @return List of connected ports
   */
  sprokit::process::ports_t port_list() const;

  virtual adapter::ports_info_t get_ports() = 0;

protected:

  std::set< sprokit::process::port_t > m_active_ports;

  // The interface queue is managed by shared pointer because it is
  // shared amongst the client API, and the process threads, so it
  // must remain active until the last user gets deleted.
  interface_ref_t  m_interface_queue;

}; // end class

} }  // end namespace

#endif /* PROCESS_ADAPTER_BASE_H */
