/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_MUTATE_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_MUTATE_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

namespace vistk
{

/**
 * \class mutate_process
 *
 * \brief A process which has an input port with the mutate flag.
 *
 * \process A process for testing mutation flags.
 *
 * \iports
 *
 * \iport{mutate} A port with the mutate flag on it.
 *
 * \reqs
 *
 * \req The \port{number} port must be connected.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT mutate_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    mutate_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~mutate_process();
  protected:
    /**
     * \brief Eats data from the input edge.
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
     * \brief Lists the ports available on the process.
     */
    ports_t _input_ports() const;
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_MUTATE_PROCESS_H
