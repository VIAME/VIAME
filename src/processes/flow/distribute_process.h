/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H
#define VISTK_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H

#include "flow-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file distribute_process.h
 *
 * \brief Declaration of the distribution process.
 */

namespace vistk
{

/**
 * \class distribute_process
 *
 * \brief A process which distributes input data along multiple output edges.
 *
 * \note Edges for a \portvar{src} may \em only be connected after the
 * \port{color_\portvar{src}} is connected to. Before this connection happens,
 * the other ports to not exist and will cause errors. In short: The first
 * connection for any \portvar{src} must be \port{color_\portvar{src}}.
 *
 * \process A process for generating numbers.
 *
 * \iports
 *
 * \iport{src_\portvar{src}} The source input \portvar{src}.
 *
 * \oports
 *
 * \oport{color_\portvar{src}} The color of the input \portvar{src}.
 * \oport{dist_\portvar{src}_\portvar{any}} A port to distribute the input
 *                                          \portvar{src} to. Data is
 *                                          distributed in ASCII-betical order.
 *
 * \reqs
 *
 * \req Each input port \port{src_\portvar{src}} must be connected.
 * \req Each output port \port{color_\portvar{res}} must be connected.
 * \req Each \portvar{res} must have at least two outputs to distribute to.
 *
 * \todo Add configuration to allow forcing a number of outputs for a source.
 * \todo Add configuration to allow same number of outputs for all sources.
 */
class VISTK_PROCESSES_FLOW_NO_EXPORT distribute_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    distribute_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~distribute_process();
  protected:
    /**
     * \brief Checks the output port connections and the configuration.
     */
    void _init();

    /**
     * \brief Distribute data between the output edges.
     */
    void _step();

    /**
     * \brief Subclass output port information.
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

#endif // VISTK_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H
