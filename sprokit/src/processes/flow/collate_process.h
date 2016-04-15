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

#ifndef SPROKIT_PROCESSES_FLOW_COLLATE_PROCESS_H
#define SPROKIT_PROCESSES_FLOW_COLLATE_PROCESS_H

#include "flow-config.h"

#include <sprokit/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file collate_process.h
 *
 * \brief Declaration of the collate process.
 */

namespace sprokit
{

/**
 * \class collate_process
 *
 * \brief A process for collating input data from multiple input edges.
 *
 * \note Edges for a \portvar{tag} may \em only be connected after the
 * \port{status/\portvar{tag}} is connected to. Before this connection happens,
 * the other ports to not exist and will cause errors. In short: The first
 * connection for any \portvar{tag} must be \port{status/\portvar{tag}}.
 *
 * \process Collate incoming data into a single stream.
 *
 * \iports
 *
 * \iport{status/\portvar{tag}} The status of the result \portvar{tag}.
 * \iport{coll/\portvar{tag}/\portvar{group}} A port to collate data for
 *                                            \portvar{tag} from. Data is
 *                                            collated from ports in
 *                                            ASCII-betical order.
 *
 * \oports
 *
 * \oport{res/\portvar{tag}} The collated result \portvar{tag}.
 *
 * \reqs
 *
 * \req Each input port \port{status/\portvar{tag}} must be connected.
 * \req Each \portvar{tag} must have at least two inputs to collate.
 * \req Each output port \port{res/\portvar{tag}} must be connected.
 *
 * \todo Add configuration to allow forcing a number of inputs for a result.
 * \todo Add configuration to allow same number of sources for all results.
 *
 * \ingroup process_flow
 */
class SPROKIT_PROCESSES_FLOW_NO_EXPORT collate_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    collate_process(kwiver::vital::config_block_sptr const& config);
    /**
     * \brief Destructor.
     */
    ~collate_process();
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
     * \brief Input port information.
     *
     * \param port The port to get information about.
     *
     * \returns Information about an input port.
     */
    port_info_t _input_port_info(port_t const& port);
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_FLOW_COLLATE_PROCESS_H
