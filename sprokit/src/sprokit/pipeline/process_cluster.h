// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PIPELINE_PROCESS_CLUSTER_H
#define SPROKIT_PIPELINE_PROCESS_CLUSTER_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include "process.h"

/**
 * \file process_cluster.h
 *
 * \brief Header for \link sprokit::process_cluster process clusters\endlink.
 */

namespace sprokit
{

/// A pre-built collection of processes.
/**
 * This class represents a set of associated processes and their
 * interconnects. An object of this type is built from a cluster
 * specification and when complete, behaves like a process.
 *
 * Clusters are built early in the loading process when the cluster
 * definitions are loaded. They can also be specified in a regular
 * pipeline description file.
 *
 * Note that a bespoke cluster can be created by using this API directly.
 *
 * \ingroup base_classes
 */
  class SPROKIT_PIPELINE_EXPORT process_cluster
    : public process
  {
  public:
    /// The processes in the cluster.
    /**
     * \returns The processes in the cluster.
     */
    processes_t processes() const;

    /// Input mappings for the cluster.
    /**
     * \returns The input mappings for the cluster.
     */
    connections_t input_mappings() const;

    /// Output mappings for the cluster.
    /**
     * \returns The output mappings for the cluster.
     */
    connections_t output_mappings() const;

    /// Internal connections between processes in the cluster.
    /**
     * \returns The internal connections between processes in the cluster.
     */
    connections_t internal_connections() const;

    /// A property which indicates that the process is a cluster of processes.
    static property_t const property_cluster;

    /// Constructor.
    /**
     * \warning Configuration errors must \em not throw exceptions here.
     *
     * \param config Contains configuration for the process.
     */
    process_cluster(kwiver::vital::config_block_sptr const& config);

    /// Destructor.
    virtual ~process_cluster();

  protected:
    /// Map a configuration value to a process.
    /**
     * This method establishes how cluster level config items are
     * mapped to the individual processes.
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

    /// Add a process to the cluster.
    /**
     * This method adds a process to the cluster. The config supplied
     * is passed directly to the process when it is created.
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

    /// Map a port to an input on the cluster.
    /**
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

    /// Map a port to an output on the cluster.
    /**
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

    /// Connect processes within the cluster.
    /**
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

    /// Pre-connection initialization for subclasses.
    /**
     *
     */
    void _configure() override;

    /// Post-connection initialization for subclasses.
    /**
     *
     */
    void _init() override;

    /// Reset logic for subclasses.
    /**
     *
     */
    void _reset() override;

    /// Finalize logic for the cluster
    /**
     *
     */
    void _finalize() override;

    /// Runtime configuration for subclasses.
    /**
     *
     *
     * \params conf The configuration block to apply.
     */
    void _reconfigure(kwiver::vital::config_block_sptr const& conf) override;

    /// Subclass property query method.
    /**
     *
     *
     * \returns Properties on the subclass.
     */
    properties_t _properties() const override;

  private:
    /// A stub implementation to ensure that clusters should not be stepped.
    /**
     *
     * \throws process_exception Always thrown since clusters should not be stepped.
     */
    void _step() final;

    class SPROKIT_PIPELINE_NO_EXPORT priv;
    std::unique_ptr<priv> d;
  };

}

#endif // SPROKIT_PIPELINE_PROCESS_CLUSTER_H
