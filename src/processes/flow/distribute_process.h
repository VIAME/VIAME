/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H
#define VISTK_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H

#include "flow-config.h"

#include <vistk/pipeline/process.h>

namespace vistk
{

/**
 * \class distribute_process
 *
 * \brief A process which generates increasing numbers within a range.
 *
 * \process A process for generating numbers.
 *
 * \iports
 *
 * \iport{src_\em src} The source input \em src.
 *
 * \oports
 *
 * \oport{color_\em src} The color of the input \em src.
 * \oport{dist_\em src _\em any} A port to distribute the input \em src to. Data
 *                               is distributed in ASCII-betical order.
 *
 * \reqs
 *
 * \req The \port{src} input must be connected.
 *
 * \todo Add configuration to allow forcing a number of outputs for a source.
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
     * \brief Subclass input connection method.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    void _connect_input_port(port_t const& port, edge_ref_t edge);
    /**
     * \brief Subclass output connection method.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    void _connect_output_port(port_t const& port, edge_ref_t edge);
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_FLOW_DISTRIBUTE_PROCESS_H
