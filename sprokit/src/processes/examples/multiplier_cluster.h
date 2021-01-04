// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PROCESSES_EXAMPLES_MULTIPLIER_CLUSTER_H
#define SPROKIT_PROCESSES_EXAMPLES_MULTIPLIER_CLUSTER_H

#include "processes_examples_export.h"

#include <sprokit/pipeline/process_cluster.h>

/**
 * \file multiplier_cluster.h
 *
 * \brief Declaration of the multiplier cluster.
 */

namespace sprokit {

/**
 * \class multiplier_cluster
 *
 * \brief A multiplier cluster.
 *
 * \process A multiplier cluster.
 *
 * \ingroup examples
 */
class PROCESSES_EXAMPLES_NO_EXPORT multiplier_cluster
  : public process_cluster
{
public:
  PLUGIN_INFO( "multiplier_cluster",
               "A constant factor multiplier cluster" );
  /**
   * \brief Constructor.
   *
   * \param config The configuration for the process.
   */
  multiplier_cluster(kwiver::vital::config_block_sptr const& config);

  /**
   * \brief Destructor.
   */
  virtual ~multiplier_cluster();

private:
  class priv;
  std::unique_ptr<priv> d;
};

}

#endif // SPROKIT_PROCESSES_EXAMPLES_MULTIPLIER_CLUSTER_H
