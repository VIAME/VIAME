/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PROCESSES_EXAMPLES_MULTIPLIER_CLUSTER_H
#define SPROKIT_PROCESSES_EXAMPLES_MULTIPLIER_CLUSTER_H

#include "examples-config.h"

#include <sprokit/pipeline/process_cluster.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file multiplier_cluster.h
 *
 * \brief Declaration of the multiplier cluster.
 */

namespace sprokit
{

/**
 * \class multiplier_cluster
 *
 * \brief A multiplier cluster.
 *
 * \process A multiplier cluster.
 *
 * \ingroup examples
 */
class SPROKIT_PROCESSES_EXAMPLES_NO_EXPORT multiplier_cluster
  : public process_cluster
{
  public:
    /**
     * \brief Constructor.
     *
     * \param config The configuration for the process.
     */
    multiplier_cluster(config_t const& config);
    /**
     * \brief Destructor.
     */
    ~multiplier_cluster();
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_MULTIPLIER_CLUSTER_H
