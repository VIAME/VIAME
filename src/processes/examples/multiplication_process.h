/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

namespace vistk
{

/**
 * \class multiplication_process
 *
 * \brief A process which multiplies incoming numbers.
 *
 * \process A process for multiplying numbers.
 *
 * \iports
 *
 * \iport{factor1} The first number to multiply.
 * \iport{factor2} The second number to multiply.
 *
 * \oports
 *
 * \oport{product} The number generated for the step.
 *
 * \reqs
 *
 * \req The \port{factor1}, \port{factor2}, and \port{product} ports must be connected.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT multiplication_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    multiplication_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~multiplication_process();
  protected:
    /**
     * \brief Multiplies numbers and outputs the result.
     */
    void _step();

    /**
     * \brief Connects an edge to an input port on the process.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    void _connect_input_port(port_t const& port, edge_t edge);

    /**
     * \brief Information about an input port on the process.
     *
     * \param port The port to return information about.
     *
     * \returns Information about the input port.
     */
    port_info_t _input_port_info(port_t const& port) const;

    /**
     * \brief Connects an edge to an output port on the process.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    void _connect_output_port(port_t const& port, edge_t edge);

    /**
     * \brief Information about an output port on the process.
     *
     * \param port The port to return information about.
     *
     * \returns Information about the output port.
     */
    port_info_t _output_port_info(port_t const& port) const;

    /**
     * \brief Lists the ports available on the process.
     */
    ports_t _input_ports() const;
    /**
     * \brief Lists the ports available on the process.
     */
    ports_t _output_ports() const;
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_MULTIPLICATION_PROCESS_H
