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
 * \brief A process which generates increasing numbers within a range.
 *
 * \process A process for generating numbers.
 *
 * \oports
 *
 * \oport{number} The number generated for the step.
 *
 * \configs
 *
 * \config{start} The start of the range.
 * \config{end} The end of the range.
 *
 * \reqs
 *
 * \req \key{start} must be less than \key{end}.
 * \req The \port{number} output must be connected to at least one edge.
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
     * \brief Connects an edge to an output port on the process.
     *
     * \param port The port to connect to.
     * \param edge The edge to connect to the port.
     */
    void _connect_output_port(port_t const& port, edge_t edge);

    /**
     * \brief Information about an output port on the process.
     *
     * \throws no_such_port_exception_exception Thrown when \p port does not exist on the process.
     *
     * \param port The port to return information about.
     *
     * \returns Information about the output port.
     */
    port_info_t _output_port_info(port_t const& port) const;

    /**
     * \brief The available configuration options for the process.
     */
    config::keys_t _available_config() const;

    /**
     * \brief Retrieve information about a configuration parameter.
     *
     * \param key The name of the configuration value.
     *
     * \returns Information about the parameter.
     */
    conf_info_t _config_info(config::key_t const& key) const;

    /**
     * \brief Lists the ports available on the process.
     */
    ports_t _output_ports() const;
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_NUMBER_PROCESS_H
