/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_PROCESS_CLUSTER_H
#define VISTK_PIPELINE_PROCESS_CLUSTER_H

#include "pipeline-config.h"

#include "process.h"

#include <boost/scoped_ptr.hpp>

/**
 * \file process_cluster.h
 *
 * \brief Header for \link vistk::process_cluster process clusters\endlink.
 */

namespace vistk
{

/**
 * \class process_cluster process_cluster.h <vistk/pipeline/process_cluster.h>
 *
 * \brief A pre-built collection of processes.
 *
 * \ingroup base_classes
 */
class VISTK_PIPELINE_EXPORT process_cluster
  : public process
{
  public:
    /**
     * \brief The processes in the cluster.
     *
     * \returns The processes in the cluster.
     */
    virtual processes_t processes() const = 0;
    /**
     * \brief Input mappings for the cluster.
     *
     * \returns The input mappings for the cluster.
     */
    virtual connections_t input_mappings() const = 0;
    /**
     * \brief Output mappings for the cluster.
     *
     * \returns The output mappings for the cluster.
     */
    virtual connections_t output_mappings() const = 0;
    /**
     * \brief Internal connections between processes in the cluster.
     *
     * \returns The internal connections between processes in the cluster.
     */
    virtual connections_t internal_connections() const = 0;

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
    process_cluster(config_t const& config);
    /**
     * \brief Destructor.
     */
    virtual ~process_cluster();

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
     * \brief Method where subclass data processing occurs.
     */
    void _step();

    /**
     * \brief Subclass property query method.
     *
     * \returns Properties on the subclass.
     */
    virtual properties_t _properties() const;
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_PIPELINE_PROCESS_CLUSTER_H
