/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_PRINT_STRING_PROCESS_H
#define VISTK_PROCESSES_EXAMPLES_PRINT_STRING_PROCESS_H

#include "examples-config.h"

#include <vistk/pipeline/process.h>

namespace vistk
{

/**
 * \class print_string_process
 *
 * \brief A process which prints incoming strings.
 *
 * \process A process for printing strings.
 *
 * \iports
 *
 * \iport{string} The source of strings to print.
 *
 * \configs
 *
 * \config{output} Where to output the strings.
 *
 * \reqs
 *
 * \req The \port{string} port must be connected.
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT print_string_process
  : public process
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config Contains config for the process.
     */
    print_string_process(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~print_string_process();
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
     * \brief Information about an input port on the process.
     *
     * \param port The port to return information about.
     *
     * \returns Information about the input port.
     */
    port_info_t _input_port_info(port_t const& port) const;

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
    ports_t _input_ports() const;
  private:
    class priv;
    boost::shared_ptr<priv> d;
};

}

#endif // VISTK_PROCESSES_EXAMPLES_PRINT_STRING_PROCESS_H
