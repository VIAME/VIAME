/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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
