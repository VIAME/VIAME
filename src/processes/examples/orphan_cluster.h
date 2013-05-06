/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_EXAMPLES_ORPHAN_CLUSTER_H
#define SPROKIT_PROCESSES_EXAMPLES_ORPHAN_CLUSTER_H

#include "examples-config.h"

#include <sprokit/pipeline/process_cluster.h>

/**
 * \file orphan_cluster.h
 *
 * \brief Declaration of the orphan cluster.
 */

namespace sprokit
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
class SPROKIT_PROCESSES_EXAMPLES_NO_EXPORT orphan_cluster
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
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_ORPHAN_CLUSTER_H
