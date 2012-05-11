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
 * \note Edges for a \portvar{tag} may \em only be connected after the
 * \port{color/\portvar{tag}} is connected to. Before this connection happens,
 * the other ports to not exist and will cause errors. In short: The first
 * connection for any \portvar{tag} must be \port{color/\portvar{tag}}.
 *
 * \note Ports sharing the same \port{\portvar{group}} string will use the same
 * coloring.
 *
 * \process A process for generating numbers.
 *
 * \iports
 *
 * \iport{src/\portvar{tag}} The source input \portvar{tag}.
 *
 * \oports
 *
 * \oport{color/\portvar{tag}} The color of the input \portvar{tag}.
 * \oport{dist/\portvar{tag}/\portvar{group}} A port to distribute the input
 *                                            \portvar{tag} to. Data is
 *                                            distributed in ASCII-betical order.
 *
 * \reqs
 *
 * \req Each input port \port{src/\portvar{tag}} must be connected.
 * \req Each output port \port{color/\portvar{res}} must be connected.
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
     * \brief Resets the process.
     */
    void _reset();

    /**
     * \brief Distribute data between the output edges.
     */
    void _step();

    /**
     * \brief Set constraints on the process.
     */
    constraints_t _constraints() const;

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
