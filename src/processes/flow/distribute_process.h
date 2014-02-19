/*ckwg +29
 * Copyright 2011-2012 by Kitware, Inc.
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

#ifndef SPROKIT_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H
#define SPROKIT_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H

#include "flow-config.h"

#include <sprokit/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file distribute_process.h
 *
 * \brief Declaration of the distribute process.
 */

namespace sprokit
{

/**
 * \class distribute_process
 *
 * \brief A process for distributing input data to multiple output edges.
 *
 * \note Edges for a \portvar{tag} may \em only be connected after the
 * \port{status/\portvar{tag}} is connected to. Before this connection happens,
 * the other ports to not exist and will cause errors. In short: The first
 * connection for any \portvar{tag} must be \port{status/\portvar{tag}}.
 *
 * \process Distribute input data among many output processes.
 *
 * \iports
 *
 * \iport{src/\portvar{tag}} The source input \portvar{tag}.
 *
 * \oports
 *
 * \oport{status/\portvar{tag}} The status of the input \portvar{tag}.
 * \oport{dist/\portvar{tag}/\portvar{group}} A port to distribute the input
 *                                            \portvar{tag} to. Data is
 *                                            distributed in ASCII-betical order.
 *
 * \reqs
 *
 * \req Each input port \port{src/\portvar{tag}} must be connected.
 * \req Each output port \port{status/\portvar{res}} must be connected.
 * \req Each \portvar{res} must have at least two outputs to distribute to.
 *
 * \todo Add configuration to allow forcing a number of outputs for a source.
 * \todo Add configuration to allow same number of outputs for all sources.
 *
 * \ingroup process_flow
 */
class SPROKIT_PROCESSES_FLOW_NO_EXPORT distribute_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    distribute_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~distribute_process();
  protected:
    /**
     * \brief Initialize the process.
     */
    void _init();

    /**
     * \brief Reset the process.
     */
    void _reset();

    /**
     * \brief Step the process.
     */
    void _step();

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
    port_info_t _output_port_info(port_t const& port);
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H
