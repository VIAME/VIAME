/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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

}; // end class frame_list_process

} }  // end namespace

#endif /* PROCESS_ADAPTER_BASE_H */
