/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_EXAMPLES_MULTIPLIER_CLUSTER_H
#define VISTK_PROCESSES_EXAMPLES_MULTIPLIER_CLUSTER_H

#include "examples-config.h"

#include <vistk/pipeline/process_cluster.h>

#include <boost/scoped_ptr.hpp>

/**
 * \file multiplier_cluster.h
 *
 * \brief Declaration of the multiplier cluster.
 */

namespace vistk
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
class VISTK_PROCESSES_EXAMPLES_NO_EXPORT multiplier_cluster
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

#endif // VISTK_PROCESSES_EXAMPLES_MULTIPLIER_CLUSTER_H
