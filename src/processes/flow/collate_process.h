/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_FLOW_COLLATE_PROCESS_H
#define VISTK_PROCESSES_FLOW_COLLATE_PROCESS_H

#include "flow-config.h"

#include <vistk/pipeline/process.h>

namespace vistk
{

/**
 * \class collate_process
 *
 * \brief A process which generates increasing numbers within a range.
 *
 * \process A process for generating numbers.
 *
 * \iports
 *
 * \iport{color_\em res} The color of the result \em res.
 * \iport{coll_\em res _\em any} A port to collate data for \em res from. Data
 *                               is collated from ports in ASCII-betical order.
 *
 * \oports
 *
 * \oport{out_\em res} The collated result \em res.
 *
 * \reqs
 *
 * \req The \port{res} input must be connected.
 *
 * \todo Add configuration to allow forcing a number of inputs for a result.
 * \todo Add configuration to allow same number of sources for all results.
 */
class VISTK_PROCESSES_FLOW_NO_EXPORT collate_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    collate_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~collate_process();
  protected:
    /**
     * \brief Checks the output port connections and the configuration.
     */
    void _init();

    /**
     * \brief Collate data from the input edges.
     */
    void _step();

    /**
     * \brief Subclass input connection method.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    void _connect_input_port(port_t const& port, edge_ref_t edge);
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_FLOW_COLLATE_PROCESS_H
