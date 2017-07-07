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
 * \brief Interface for input_adapter class
 */

#ifndef KWIVER_INPUT_ADAPTER_H
#define KWIVER_INPUT_ADAPTER_H

#include <sprokit/processes/adapters/kwiver_adapter_export.h>

#include "adapter_types.h"
#include "adapter_data_set.h"


namespace kwiver {

class input_adapter_process;

// -----------------------------------------------------------------
/**
 * @brief Input adapter class that provides input to pipeline.
 *
 * This class represents a user interface to the input end of a
 * sprokit pipeline. An object of this class attaches to the
 * input_adapter_process in a pipeline and provides an API to
 * communicate with that process which is running in another thread.
 */
class KWIVER_ADAPTER_EXPORT input_adapter
{
public:
  input_adapter();
  virtual ~input_adapter();

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
   * This method returns the list of input ports that are connected
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
   * @brief Send data set to input adapter process.
   *
   * The specified data set is sent to the input adapter process that
   * is currently connected to this object.
   *
   * @param dat Data set to send.
   */
  void send( kwiver::adapter::adapter_data_set_t dat );

  /**
   * @brief Is interface queue full?
   *
   * This method checks to see if the input adapter process can accept
   * more data.
   *
   * @return \b true if interface queue is full and thread would wait
   * for send().
   */
  bool full() const;

private:
  kwiver::input_adapter_process* m_process;
  kwiver::adapter::interface_ref_t m_interface_queue;
}; // end class input_adapter

} // end namespace

#endif // KWIVER_INPUT_ADAPTER_H
