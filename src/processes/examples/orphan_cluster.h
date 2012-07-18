/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_ORPHAN_CLUSTER_H
#define VISTK_PROCESSES_EXAMPLES_ORPHAN_CLUSTER_H

#include "examples-config.h"

#include <vistk/pipeline/process_cluster.h>

/**
 * \file orphan_cluster.h
 *
 * \brief Declaration of the orphan cluster.
 */

namespace vistk
{

/**
 * \class orphan_cluster
 *
 * \brief A no-op cluster.
 *
 * \process A no-op cluster.
 *
 * \ingroup examples
 */
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT orphan_cluster
  : public process_cluster
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    orphan_cluster(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~orphan_cluster();

    /**
     * \brief The processes in the cluster.
     *
     * \returns The processes in the cluster.
     */
    vistk::processes_t processes() const;
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
};

}

#endif // VISTK_PROCESSES_EXAMPLES_ORPHAN_CLUSTER_H
