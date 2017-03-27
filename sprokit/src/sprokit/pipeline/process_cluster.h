/*ckwg +29
 * Copyright 2012-2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SPROKIT_PIPELINE_PROCESS_CLUSTER_H
#define SPROKIT_PIPELINE_PROCESS_CLUSTER_H

#include "pipeline-config.h"

#include "process.h"

/**
 * \file process_cluster.h
 *
 * \brief Header for \link sprokit::process_cluster process clusters\endlink.
 */

namespace sprokit
{

/**
 * \brief A pre-built collection of processes.
 *
 * This class represents a set of associated processes and their
 * interconnects. An object of this type is built from a cluster
 * specification and when complete, behaves like a process.
 *
 * Clusters are built early in the loading process when the cluster
 * definitions are loaded. They can also be specified in a regular
 * pipeline description file.
 *
 * \ingroup base_classes
 */
class SPROKIT_PIPELINE_EXPORT process_cluster
  : public process
{
  public:
    /**
     * \brief The processes in the cluster.
     *
     * \returns The processes in the cluster.
     */
    processes_t processes() const;

    /**
     * \brief Input mappings for the cluster.
     *
     * \returns The input mappings for the cluster.
     */
    connections_t input_mappings() const;

    /**
     * \brief Output mappings for the cluster.
     *
     * \returns The output mappings for the cluster.
     */
    connections_t output_mappings() const;

    /**
     * \brief Internal connections between processes in the cluster.
     *
     * \returns The internal connections between processes in the cluster.
     */
    connections_t internal_connections() const;

    /// A property which indicates that the process is a cluster of processes.
    static property_t const property_cluster;

  protected:
    /**
     * \brief Constructor.
     *
     * \warning Configuration errors must \em not throw exceptions here.
     *
     * \param config Contains configuration for the process.
     */
    process_cluster(kwiver::vital::config_block_sptr const& config);

    /**
     * \brief Destructor.
     */
    virtual ~process_cluster();

    /**
     * \brief Map a configuration value to a process.
     *
     * \throws mapping_after_process_exception Thrown when a process named \p name_ already exists.
     *
     * \param key The key on the cluster.
     * \param name_ The process to map the configuration to.
     * \param mapped_key The key to map the configuration to.
     */
    void map_config(kwiver::vital::config_block_key_t const& key,
                    name_t const& name_,
                    kwiver::vital::config_block_key_t const& mapped_key);

    /**
     * \brief Add a process to the cluster.
     *
     * \throws duplicate_process_name_exception Thrown when a process named \p name_ already exists.
     *
     * \param name_ The name of the process.
     * \param type_ The type of the process.
     * \param config The base configuration to use.
     */
    void add_process(name_t const& name_,
                     type_t const& type_,
                     kwiver::vital::config_block_sptr const& config = kwiver::vital::config_block::empty_config());

    /**
     * \brief Map a port to an input on the cluster.
     *
     * \throws no_such_process_exception Thrown when \p name_ does not exist in the cluster.
     * \throws no_such_port_exception Thrown when the process \p name_ does not have an input port \p port.
     *
     * \param port The port on the cluster.
     * \param name_ The name of the process to map the input to.
     * \param mapped_port The port on the process to map the input to.
     */
    void map_input(port_t const& port,
                   name_t const& name_,
                   port_t const& mapped_port);

    /**
     * \brief Map a port to an output on the cluster.
     *
     * \throws no_such_process_exception Thrown when \p name_ does not exist in the cluster.
     * \throws no_such_port_exception Thrown when the process \p name_ does not have an output port \p port.
     *
     * \param port The port on the cluster.
     * \param name_ The name of the process to map the output to.
     * \param mapped_port The port on the process to map the output to.
     */
    void map_output(port_t const& port,
                    name_t const& name_,
                    port_t const& mapped_port);

    /**
     * \brief Connect processes within the cluster.
     *
     * \throws no_such_process_exception Thrown when either \p upstream_name or \p downstream_name do not exist in the cluster.
     * \throws no_such_port_exception Thrown when a port requested for connection does not exist.
     *
     * \param upstream_name The upstream process name.
     * \param upstream_port The upstream process port.
     * \param downstream_name The downstream process name.
     * \param downstream_port The downstream process port.
     */
    void connect(name_t const& upstream_name, port_t const& upstream_port,
                 name_t const& downstream_name, port_t const& downstream_port);

    /**
     * \brief Pre-connection initialization for subclasses.
     */
    void _configure();

    /**
     * \brief Post-connection initialization for subclasses.
     */
    void _init();

    /**
     * \brief Reset logic for subclasses.
     */
    void _reset();

    /**
     * \brief A stub implementation to ensure that clusters should not be stepped.
     *
     * \throws process_exception Always thrown since clusters should not be stepped.
     */
    void _step();

    /**
     * \brief Runtime configuration for subclasses.
     *
     * \params conf The configuration block to apply.
     */
    virtual void _reconfigure(kwiver::vital::config_block_sptr const& conf);

    /**
     * \brief Subclass property query method.
     *
     * \returns Properties on the subclass.
     */
    virtual properties_t _properties() const;

  private:
    class SPROKIT_PIPELINE_NO_EXPORT priv;
    std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PIPELINE_PROCESS_CLUSTER_H
