/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

namespace vistk
{

/**
 * \class print_number_process
 *
 * \brief A process which prints incoming numbers.
 *
 * \process A process for printing numbers.
 *
 * \iports
 *
 * \iport{number} The source of numbers to print.
 *
 * \configs
 *
 * \config{output} Where to output the numbers.
 *
 * \reqs
 *
 * \req The \port{number} port must be connected.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT print_number_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    print_number_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~print_number_process();

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
     * \brief Prints numbers to the output stream.
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
     * \brief The type of data that is available on an input port.
     *
     * \param port The port to return the type of.
     *
     * \returns The type of data expected.
     */
    port_type_t _input_port_type(port_t const& port) const;
    /**
     * \brief Describe input ports on the process.
     *
     * \param port The port to describe.
     *
     * \returns A description of the port.
     */
    port_description_t _input_port_description(port_t const& port) const;

    /**
     * \brief The available configuration options for the process.
     */
    config::keys_t _available_config() const;
    /**
     * \brief Request the default value for a configuration.
     *
     * \param key The name of the configuration value.
     */
    config::value_t _config_default(config::key_t const& key) const;
    /**
     * \brief Request available configuration options for the process.
     *
     * \param key The name of the configuration value to describe.
     */
    config::description_t _config_description(config::key_t const& key) const;

    /**
     * \brief Lists the ports available on the process.
     */
    ports_t _input_ports() const;
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_PRINT_NUMBER_PROCESS_H
