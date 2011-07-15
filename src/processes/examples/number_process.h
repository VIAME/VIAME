/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_NUMBER_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_NUMBER_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

namespace vistk
{

/**
 * \class number_process
 *
 * \brief A connection between two process ports which can carry data.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT number_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    number_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~number_process();

    /**
     * \brief Returns the type of the process.
     */
    process_registry::type_t type() const;
  protected:
    /**
     * \brief Checks the output port connections and the configuration.
     */
    void _init();

    /**
     * \brief Pushes a new number through the output edge.
     */
    void _step();

    /**
     * \brief Connects an edge to the output port.
     */
    void _connect_output_port(port_t const& port, edge_t edge);

    /**
     * \brief Lists the ports available on the process.
     */
    ports_t _output_ports() const;
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

} // end namespace vistk

#endif // VISTK_PROCESSES_EXAMPLES_NUMBER_PROCESS_H
