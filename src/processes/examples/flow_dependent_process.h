/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_FLOW_DEPENDENT_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_FLOW_DEPENDENT_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file flow_dependent_process.h
 *
 * \brief Declaration of the flow dependent process.
 */

namespace vistk
{

/**
 * \class flow_dependent_process
 *
 * \brief A process which has flow dependent ports.
 *
 * \process A process with flow dependent ports.
 *
 * \configs
 *
 * \config{reject} Whether to reject the set type or not.
 *
 * \iports
 *
 * \iport{input} A flow dependent input port.
 *
 * \oports
 *
 * \oport{output} A flow dependent output port.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT flow_dependent_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    flow_dependent_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~flow_dependent_process();
  protected:
    /**
     * \brief Resets the process.
     */
    void _reset();
    /**
     * \brief Pushes a new number through the output edge.
     */
    void _step();
    /**
     * \brief Sets the type for an input port.
     *
     * \param port The name of the port.
     * \param type The type of the connected port.
     *
     * \returns True if the type can work, false otherwise.
     */
    bool _set_input_port_type(port_t const& port, port_type_t const& new_type);
    /**
     * \brief Sets the type for an output port.
     *
     * \param port The name of the port.
     * \param type The type of the connected port.
     *
     * \returns True if the type can work, false otherwise.
     */
    bool _set_output_port_type(port_t const& port, port_type_t const& new_type);
  private:
    void make_ports();

    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_FLOW_DEPENDENT_PROCESS_H
