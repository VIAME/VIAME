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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \file mux_process.h
 *
 * \brief Declaration of the mux process.
 */

#ifndef SPROKIT_PROCESSES_FLOW_MUX_PROCESS_H
#define SPROKIT_PROCESSES_FLOW_MUX_PROCESS_H

#include "processes_flow_export.h"

#include <sprokit/pipeline/process.h>


namespace sprokit {

class PROCESSES_FLOW_NO_EXPORT mux_process
  : public process
{
public:
  PLUGIN_INFO( "multiplexer",
               "Multiplex incoming data into a single stream.\n\n"
               "A mux operation reads input from a group of input ports and serializes "
               "that data to a single output port. The ports in a group are read in "
               "ASCII-betical order over the third port name component (<item>). "
               "This mux process can handle multiple collation operations. Each set "
               "of input ports is identified by a unique group name. "
               "\n\n"
               "Input ports are dynamically created as needed. Port names have the "
               "format 'in/<group>/<item>'. The multiplexed result of a group of input ports "
               "is placed on the output port named 'out/<group>'."
               "\n\n"
               "Each <group> must have at least two inputs to mux and "
               "each output port 'out/<group>' must be connected. "
               "\n\n"
               "This process automatically makes the input and output types for "
               "each <group> the same based on the type of the port that is first "
               "connected."
               "\n\nExample:\n"
               "process mux :: multiplexer\n"
               "\n"
               "# -- Connect mux set 'foo'"
               "# All inputs for a group must have the same type\n"
               "connect foo_1.out       to  mux.in/foo/A\n"
               "connect foo_2.out       to  mux.in/foo/B\n"
               "\n"
               "# Create another group for the timestamp outputs.\n"
               "# For convenience the group name is 'timestamp'\n"
               "connect foo_1.timestamp   to mux.in/timestamp/A\n"
               "connect foo_2.timestamp   to mux.in/timestamp/B\n"
               "\n"
               "connect mux.out/foo         to bar.input # connect output of group\n"
               "connect mux.out/timestamp   to bar.timestamp # connect output of group"
    )

  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  mux_process( kwiver::vital::config_block_sptr const& config );

  /**
   * \brief Destructor.
   */
  virtual ~mux_process();


protected:
  void _configure() override;
  void _init() override;
  void _reset() override;
  void _step() override;
  properties_t _properties() const override;

  void input_port_undefined( port_t const& port ) override;


private:
  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // SPROKIT_PROCESSES_FLOW_MUX_PROCESS_H
